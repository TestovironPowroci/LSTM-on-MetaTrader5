from pyautogui import size, position, moveTo, click
from time import sleep
from torch import argmax, zeros, Tensor, load, optim
import torch.nn as nn
import numpy as np
import os.path
import ctypes
import ctypes.wintypes


# parameters here must be the same as in training version of LSTM network
bidirectional = True
seq_lengths = 18
input_size = 4
hidden_size = 8
n_layers = 3
output_size = 3

screenWidth, screenHeight = size()
currentMouseX, currentMouseY = position()

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_READ = 0x0010
PROCESS_VM_WRITE = 0x0020

MAX_PATH = 260


class ReadWriteMemory:

    def GetProcessIdByName(self, pName):
        if pName.endswith('.exe'):
            pass
        else:
            pName = pName + '.exe'

        ProcessIds, BytesReturned = self.EnumProcesses()

        for index in list(range(int(BytesReturned / ctypes.sizeof(ctypes.wintypes.DWORD)))):
            ProcessId = ProcessIds[index]
            hProcess = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, ProcessId)
            if hProcess:
                ImageFileName = (ctypes.c_char * MAX_PATH)()
                if ctypes.windll.psapi.GetProcessImageFileNameA(hProcess, ImageFileName, MAX_PATH) > 0:
                    filename = os.path.basename(ImageFileName.value)
                    if filename.decode('utf-8') == pName:
                        return ProcessId
                self.CloseHandle(hProcess)

    def EnumProcesses(self):
        count = 32
        while True:
            ProcessIds = (ctypes.wintypes.DWORD * count)()
            cb = ctypes.sizeof(ProcessIds)
            BytesReturned = ctypes.wintypes.DWORD()
            if ctypes.windll.Psapi.EnumProcesses(ctypes.byref(ProcessIds), cb, ctypes.byref(BytesReturned)):
                if BytesReturned.value < cb:
                    return ProcessIds, BytesReturned.value
                else:
                    count *= 2
            else:
                return None

    def OpenProcess(self, dwProcessId):
        dwDesiredAccess = (PROCESS_QUERY_INFORMATION |
                           PROCESS_VM_OPERATION |
                           PROCESS_VM_READ | PROCESS_VM_WRITE)
        bInheritHandle = False
        hProcess = ctypes.windll.kernel32.OpenProcess(
            dwDesiredAccess,
            bInheritHandle,
            dwProcessId
        )
        if hProcess:
            return hProcess
        else:
            return None

    def CloseHandle(self, hProcess):
        ctypes.windll.kernel32.CloseHandle(hProcess)
        return self.GetLastError()

    def GetLastError(self):
        return ctypes.windll.kernel32.GetLastError()

    def getPointer(self, hProcess, lpBaseAddress, offsets):
        pointer = self.ReadProcessMemory2(hProcess, lpBaseAddress)
        if offsets == None:
            return lpBaseAddress
        elif len(offsets) == 1:
            temp = int(str(pointer), 0) + int(str(offsets[0]), 0)
            return temp
        else:
            count = len(offsets)
            for i in offsets:
                count -= 1
                temp = int(str(pointer), 0) + int(str(i), 0)
                pointer = self.ReadProcessMemory2(hProcess, temp)
                if count == 1:
                    break
            return pointer

    def ReadProcessMemory(self, hProcess, lpBaseAddress):
        try:
            lpBaseAddress = lpBaseAddress
            ReadBuffer = ctypes.c_double()
            lpBuffer = ctypes.byref(ReadBuffer)
            nSize = ctypes.sizeof(ReadBuffer)
            lpNumberOfBytesRead = ctypes.c_char_p(0)

            ctypes.windll.kernel32.ReadProcessMemory(
                hProcess,
                lpBaseAddress,
                lpBuffer,
                nSize,
                lpNumberOfBytesRead
            )
            return ReadBuffer.value

        except (BufferError, ValueError, TypeError):
            self.CloseHandle(hProcess)
            e = 'Handle Closed, Error', hProcess, self.GetLastError()
            return e

    def ReadProcessMemory2(self, hProcess, lpBaseAddress):
        try:
            lpBaseAddress = lpBaseAddress
            ReadBuffer = ctypes.c_double()
            lpBuffer = ctypes.byref(ReadBuffer)
            nSize = ctypes.sizeof(ReadBuffer)
            lpNumberOfBytesRead = ctypes.c_char_p(0)

            ctypes.windll.kernel32.ReadProcessMemory(
                hProcess,
                lpBaseAddress,
                lpBuffer,
                nSize,
                lpNumberOfBytesRead
            )
            return ReadBuffer.value
        except (BufferError, ValueError, TypeError):
            self.CloseHandle(hProcess)
            e = 'Handle Closed, Error', hProcess, self.GetLastError()
            return e




def play_episode(s):

    c = classifier(s)
    a = argmax(c)

    if a == 2:
        print("Buy")
        moveTo(262, 457)
        click()
        moveTo(164, 155)
        click()
        moveTo(672, 181)
        click()


    elif a == 1:
        print("Sell")
        moveTo(262, 457)
        click()
        moveTo(40, 153)
        click()
        moveTo(672, 181)
        click()



    else:

        print("No decision")





class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_size, hidden_size, n_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = zeros(self.n_layers, x.size(0), self.hidden_size)
        c0 = zeros(self.n_layers, x.size(0), self.hidden_size)
        return [t for t in (h0, c0)]

        return y


classifier = LSTMClassifier(input_size, hidden_size, n_layers, output_size)
opt = optim.Adam(classifier.parameters())
criterion = nn.CrossEntropyLoss()

classifier.load_state_dict(load('path/classifier.pth',map_location='cpu'))
opt.load_state_dict(load('path/optimizer.pth',map_location='cpu'))
classifier.eval()


if __name__ == '__main__':

    while True:
        sleep(1)

        data = np.loadtxt(
            'path/date.csv',
            delimiter=',', dtype=np.float32, encoding='utf-16')
        LSTMnetwork = Tensor(data).view(1, seq_lengths, input_size)
        rwm = ReadWriteMemory()
        ProcID = rwm.GetProcessIdByName('terminal64.exe')
        hProcess = rwm.OpenProcess(ProcID)
        pos = rwm.ReadProcessMemory(hProcess, int(0,16))
        print(pos)

        if pos == 12332100:
            play_episode(LSTMnetwork)
            sleep(3)


