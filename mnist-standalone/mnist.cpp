#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>
#include <memory>
#include <numeric>
#include "buffers.h"
#include <cmath>
#include <iomanip>

using namespace nvinfer1;
using namespace nvonnxparser;

// Simple Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        // suppress info-level messages
        std::cout << msg << std::endl;
    }
} gLogger;

struct ModelParams {
    int32_t batchSize{1};              //!< Number of inputs in a batch
    int32_t dlaCore{-1};               //!< Specify the DLA core to run network on.
    bool int8{false};                  //!< Allow runnning the network in Int8 mode.
    bool fp16{false};                  //!< Allow running the network in FP16 mode.
    bool bf16{false};                  //!< Allow running the network in BF16 mode.
    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    std::string onnxFileName; //!< Filename of ONNX file of a network
};


ModelParams initializeModelParams(int argc, char *argv[]) {
    ModelParams params;
    params.dataDirs.push_back("data/mnist/");
    params.dataDirs.push_back("data/samples/mnist/");
    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    //params.dlaCore = args.useDLACore;
    params.int8 = false; //args.runInInt8;
    params.fp16 = false; //args.runInFp16;
    params.bf16 = false; //args.runInBf16;

    return params;
}

//! Locate path to file, given its filename or filepath suffix and possible dirs it might lie in.
//! Function will also walk back MAX_DEPTH dirs from CWD to check for such a file path.
inline std::string locateFile(
    const std::string& filepathSuffix, const std::vector<std::string>& directories, bool reportError = true)
{
    const int MAX_DEPTH{10};
    bool found{false};
    std::string filepath;

    for (auto& dir : directories){
        if (!dir.empty() && dir.back() != '/') {

            filepath = dir + "/" + filepathSuffix;
        } else {
            filepath = dir + filepathSuffix;
        }

        for (int i = 0; i < MAX_DEPTH && !found; i++) {
            const std::ifstream checkFile(filepath);
            found = checkFile.is_open();
            if (found){
                break;
            }

            filepath = "../" + filepath; // Try again in parent dir
        }

        if (found){
            break;
        }

        filepath.clear();
    }

    // Could not find the file
    if (filepath.empty()){
        const std::string dirList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
            [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
        std::cout << "Could not find " << filepathSuffix << " in data directories:\n\t" << dirList << std::endl;

        if (reportError){
            std::cout << "&&&& FAILED" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return filepath;
}

inline void readPGMFile(const std::string& fileName, uint8_t* buffer, int32_t inH, int32_t inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    SAFE_ASSERT(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, w, h, max;
    infile >> magic >> w >> h >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

struct InferDeleter {
    template <typename T>
    void operator()(T* obj) const{
        delete obj;
    }
};

/// Tag type for binary construction.
struct binary_t
{
};

#define HALF_NOEXCEPT noexcept
#define HALF_NOTHROW noexcept

#define HALF_CONSTEXPR			constexpr
#define HALF_CONSTEXPR_CONST	constexpr

/// Tag for binary construction.
HALF_CONSTEXPR_CONST binary_t binary = binary_t();

/// Temporary half-precision expression.
/// This class represents a half-precision expression which just stores a single-precision value internally.
struct expr
{
    /// Conversion constructor.
    /// \param f single-precision value to convert
    explicit HALF_CONSTEXPR expr(float f) HALF_NOEXCEPT : value_(f) {}

    /// Conversion to single-precision.
    /// \return single precision value representing expression value
    HALF_CONSTEXPR operator float() const HALF_NOEXCEPT
    {
        return value_;
    }

private:
    /// Internal expression value stored in single-precision.
    float value_;
};

static expr exp(float arg){
    return expr(std::exp(arg));
}

class Inference {
public:
    bool Build(ModelParams& params) {
        mParams = params;
        auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
        if (!builder){
            return false;
        }

        auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(0));
        if (!network) {
            return false;
        }

        auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            return false;
        }

        // Configure
        if (mParams.fp16) {
            config->setFlag(BuilderFlag::kFP16);
        }
        if (mParams.bf16) {
            config->setFlag(BuilderFlag::kBF16);
        }
        if (mParams.int8) {
            config->setFlag(BuilderFlag::kINT8);
            //samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
        }


        auto parser = std::unique_ptr<IParser>(createParser(*network, gLogger));
        if (!parser) {
            return false;
        }

        auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), 1);
        if (!parsed) {
            return false;
        }        
        /*
        auto constructed = constructNetwork(builder, network, config, parser);
        if (!constructed) {
            return false;
        }

        // CUDA stream used for profiling by the builder.
        auto profileStream = samplesCommon::makeCudaStream();
        if (!profileStream) {
            return false;
        }
        config->setProfileStream(*profileStream);
        */
       std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan){
            return false;
        }

        mRuntime = std::shared_ptr<IRuntime>(createInferRuntime(gLogger));   
        if (mRuntime == nullptr) {
            return false;
        } 

        mEngine = std::shared_ptr<ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
        if (!mEngine) {
            return false;
        }
        mInputDims = network->getInput(0)->getDimensions();
        mOutputDims = network->getOutput(0)->getDimensions();
        
        return true;
    }


    bool Infer() {
        // Create RAII buffer manager object
        BufferManager buffers(mEngine);

        auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        if (!context) {
            return false;
        }

        for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++) {
            auto const name = mEngine->getIOTensorName(i);
            context->setTensorAddress(name, buffers.getDeviceBuffer(name));
        }

        // Read the input data into the managed buffers
        ASSERT(mParams.inputTensorNames.size() == 1);
        if (!processInput(buffers)) {
            return false;
        }

        // Memcpy from host input buffers to device input buffers
        buffers.copyInputToDevice();

        bool status = context->executeV2(buffers.getDeviceBindings().data());
        if (!status){
            return false;
        }

        // Memcpy from device output buffers to host output buffers
        buffers.copyOutputToHost();

        // Verify results
        if (!verifyOutput(buffers)){
            return false;
        }

        return true;
    }


    //!
    //! \brief Reads the input and stores the result in a managed buffer
    //!
    bool processInput(const BufferManager& buffers) {
        const int inputH = mInputDims.d[2];
        const int inputW = mInputDims.d[3];

        // Read a random digit file
        srand(unsigned(time(nullptr)));
        std::vector<uint8_t> fileData(inputH * inputW);
        mNumber = rand() % 10;
        readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

        // Print an ascii representation
        /*
        sample::gLogInfo << "Input:" << std::endl;
        for (int i = 0; i < inputH * inputW; i++) {
            sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
        }
        sample::gLogInfo << std::endl;
        */
        float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
        for (int i = 0; i < inputH * inputW; i++) {
            hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
        }

        return true;
    }

    //!
    //! \brief Classifies digits and verify result
    //!
    //! \return whether the classification output matches expectations
    //!
    bool verifyOutput(const BufferManager& buffers)
    {
        const int outputSize = mOutputDims.d[1];
        float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
        float val{0.0F};
        int idx{0};

        // Calculate Softmax
        float sum{0.0F};
        for (int i = 0; i < outputSize; i++) {
            output[i] = exp(output[i]);
            sum += output[i];
        }

        // sample::gLogInfo << "Output:" << std::endl;
        for (int i = 0; i < outputSize; i++) {
            output[i] /= sum;
            val = std::max(val, output[i]);
            if (val == output[i]) {
                idx = i;
            }
            /*
            sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                            << " "
                            << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5F)), '*')
                            << std::endl;
            */
            std::cout << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                            << " "
                            << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5F)), '*')
                            << std::endl;
        }
        //sample::gLogInfo << std::endl;

        return idx == mNumber && val > 0.9F;
    }

public:
    Inference(): mRuntime(nullptr) {}  
    std::shared_ptr<IRuntime> mRuntime;
    std::shared_ptr<ICudaEngine> mEngine;
    Dims mInputDims;  //!< The dimensions of the input to the network.
    Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};    

    ModelParams mParams;
};

int main(int argc, char *argv[]) {

    std::string engine_file = "Example.engine";
    Inference inference;
    // Create a TensorRT runtime
    auto params = initializeModelParams(argc, argv);
    inference.Build(params);
    auto res = inference.Infer();
}