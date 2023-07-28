# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2023 Fujitsu Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import traceback
from collections import deque
import threading


class MuteStdout:
    """Suppress message emission to stdout."""
    def __init__(self, debug=False):
        self.org_stdout = sys.stdout
        self.out_buffer = DevNull(debug=debug)
        self.debug = debug
    def __enter__(self):
        sys.stdout = self.out_buffer
        return self
    def __exit__(self, ex_type, ex_value, tracebac):
        sys.stdout = self.org_stdout
        if ex_value is not None:
            if self.debug:
                print(f'[OUTPUT start] thread{threading.current_thread().name}({threading.current_thread().ident}) queue={id(self.out_buffer.queue)} len={len(self.out_buffer.queue)}')
            for message in self.out_buffer.queue:
                sys.stdout.write(message)
            if self.debug:
                print('[OUTPUT end]', flush=True)
        if tracebac is not None:
            traceback.print_exception(ex_type, ex_value, tracebac)
        if ex_value is not None:
            raise ex_value


class DevNull:
    """Output stream to /dev/null."""
    def __init__(self, debug=False):
        self.debug = debug
        if self.debug:
            print('***DEVNULL INITIALIZED **************', flush=True, file=sys.stderr)
        self.queue = deque()
    def write(self, message):
        self.queue.append(message)
        if self.debug:
            print(f'thread{threading.current_thread().name}({threading.current_thread().ident}) queue={id(self.queue)} len={len(self.queue)}', end='   \r', file=sys.stderr)
            # print(f'thread{threading.current_thread().name}({threading.current_thread().ident}) queue={id(self.queue)} len={len(self.queue)} {message}', end='', file=sys.stderr)
    def flush(self):
        pass
