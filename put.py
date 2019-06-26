import paramiko
import os
from os.path import join as pjoin
from time import sleep

paramiko.util.log_to_file('/home/b04112/paramiko.log')

# Open a transport
host = "140.112.175.124"
port = 22
transport = paramiko.Transport((host, port))

# Auth
password = "tmp1234"
username = "b04112"
transport.connect(username = username, password = password)

sftp = paramiko.SFTPClient.from_transport(transport)

localBase = '/home/b04112/aca-training'
remoteBase = '/home/b04112/newDat'

dirs = ['_log', '_logh', '_parth', '_partition', '_result', '_single']
while True:
    print('putting files...', end='')
    for d in dirs:
        files = os.listdir(pjoin(localBase, d))
        files = filter(lambda x: ('vgg' in x and ('3000' in x or '6000' in x)) or 'resnext' in x, files)
        for f in files:
            localPath = pjoin(localBase, d, f)
            remotePath = pjoin(remoteBase, d, f)
            sftp.put(localPath, remotePath)
    print('done')
    sleep(600)


sftp.close()
transport.close()
