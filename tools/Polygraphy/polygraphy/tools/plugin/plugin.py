#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from polygraphy.tools.base import Tool
from polygraphy.tools.plugin.subtool import Match, ListPlugins, Replace


class Plugin(Tool):
    """
    Plugin related operations on an onnx model.
    """

    def __init__(self):
        super().__init__("plugin")

    def get_subtools_impl(self):
        return "Plugin Subtools", [
            Match(),
            ListPlugins(),
            Replace(),
        ]
