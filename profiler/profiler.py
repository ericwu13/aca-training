import sys
import simplejson
import numpy as np

class Profiler:
    def __init__(self, jsonPath, layerName=None):
        f = open(jsonPath)
        self.data = simplejson.load(f)
        self.name = layerName
        self.called = False
        f.close()
        for event in self.data['traceEvents']:
            if 'args' in event and 'name' in event['args']:
                if 'stream:all Compute' in event['args']['name']:
                    # print(event['args']['name'])
                    self.computePID = event['pid']
                elif 'device:GPU:0 Compute' in event['args']['name']:
                    # print(event['args']['name'])
                    self.loadPID = event['pid']

    def getExecution(self):
        t = []
        for event in self.data['traceEvents']:
            if 'args' in event and event['pid'] == self.computePID:
                name = event['args']['name']
                if 'edge' not in name and 'Compute' not in name:
                    # print("{}, {} us".format(event['args']['name'], event['ts']))
                    t.append(event['ts'])
                    t.append(event['ts']+event['dur'])
        # print("{} us".format(t))
        self.computeStart = min(t)
        self.computeStop = max(t)
        return max(0, max(t) - min(t))

    def getOverHead(self):
        t = []
        for event in self.data['traceEvents']:
            if event['pid'] == self.computePID:
                if 'ts' in event and 'dur' in event:
                    t.append(event['ts'])
                    t.append(event['ts']+event['dur'])
        computeStart = min(t)
        computeStop = max(t)
        # print(computeStop, computeStart)
        return max(0, max(t) - min(t))

    def getLoading(self):
        self.called = True
        t = 0
        for event in self.data['traceEvents']:
            if 'args' in event and event['pid'] == self.loadPID:
                name = event['args']['name'].split('/')
                # print(name)
                if name[0] == self.name and name[-1] == event['args']['op']:
                    t += event['dur']
                    # print(event['args']['name'])
        # print("{} us".format(t))
        return t

    def getMemcpy(self):
        if not self.called:
            self.getExecution()
        th2d = []
        td2h = []
        for event in self.data['traceEvents']:
            if 'args' in event:
                if 'name' in event['args']:
                    name = event['args']['name'].split('/')
                    if 'MemcpyHtoD' in name[-1] and 'dur' in event:
                        th2d.append(event['ts'])
                        th2d.append(event['ts'] + event['dur'])
                        # print("{}, {} us".format(event['args']['name'], event['dur']))
                        # print("{} us".format(th2d))
                    elif 'MemcpyDtoH' in name[-1] and 'dur' in event:
                        td2h.append(event['ts'])
                        td2h.append(event['ts'] + event['dur'])
                        # print("{}, {} us".format(event['args']['name'], event['dur']))
                        # print("{} us".format(td2h))
        if(len(th2d)):
            th2d = self.computeStart - min(th2d)
        else:
            th2d = 0
        if(len(td2h)):
            td2h = max(td2h) - self.computeStop
        else:
            td2h = 0
        self.called = False
        return [max(0, th2d), max(0, td2h)]

    def getTime(self, type):
        if type == 'exec':
            return self.getExecution()
        elif type == 'load':
            return self.getLoading()
        elif type == 'memcpy':
            return self.getMemcpy()
        elif type == 'overhead':
            return self.getOverHead()

class Collector:
    def __init__(self, n, m = 4):
        self.numLayer = n
        self.numMachine = m
        self.activation = []
        self.weight = []
        self.avgExec = []
        self.avgComm = []
        self.execution = []
        self.communication = []

    def collectProfile(self, execTime, commTime):
        self.execution.append(execTime)
        self.communication.append(commTime)

    def collectSize(self, actiSize, weightSize):
        self.activation.append(actiSize)
        self.weight.append(weightSize)

    def reset(self):
        self.avgExec.append(self.execution)
        self.avgComm.append(self.communication)

        self.execution = []
        self.communication = []

    def dump(self, fileName):
        self.avgExec = (np.mean(self.avgExec, axis=0, dtype=int))
        self.avgComm = (np.mean(self.avgComm, axis=0, dtype=int))
        print(self.avgExec)
        print(self.avgComm)


        assert(len(self.avgExec) == self.numLayer and
               len(self.avgComm) == self.numLayer and
               len(self.activation) == self.numLayer and
               len(self.weight) == self.numLayer )
        with open(fileName, 'w') as f:
            f.write("[N]\n")
            f.write(str(self.numLayer))
            f.write("\n")

            f.write("[M]\n")
            f.write(str(self.numMachine))
            f.write("\n")

            f.write("[Tl]\n")
            for t in self.avgExec:
                f.write("{} ".format(int(t)))
            f.write("\n")

            f.write("[Cl]\n")
            for c in self.avgComm:
                f.write("{} ".format(int(c)))
            f.write("\n")

            f.write("[al]\n")
            for a in self.activation:
                f.write("{} ".format(a))
            f.write("\n")

            f.write("[wl]\n")
            for w in self.weight:
                f.write("{} ".format(w))
            f.write("\n")

if __name__ == '__main__':
    profiler = Profiler(sys.argv[1], sys.argv[2])
    print(profiler.getTime(sys.argv[3]))
