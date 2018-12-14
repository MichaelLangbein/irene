import os
import bz2
import tarfile
import datetime as dt
from time import sleep
import urllib.request
from ftplib import FTP
import ftplib


def extract(path, fileName):
    fullName = path + fileName
    if not os.path.isfile(fullName):
        raise Exception("File {} does not exist!".format(fullName))
    if (fullName.endswith("tar.gz")):
        with tarfile.open(fullName, "r:gz") as tar:
            tar.extractall(path)
    elif (fullName.endswith("tar")):
        with tarfile.open(fullName, "r:") as tar:
            tar.extractall(path)
    elif(fullName.endswith(".bz2")):
        extractedName = fullName[:-4]
        with bz2.BZ2File(fullName) as zipFile: # open the file
            with open(extractedName, "wb") as file:
                data = zipFile.read() # get the decompressed data
                file.write(data)



def httpDownloadFile(serverName, path, fileName, targetDir):
    """ 
    >>> httpDownloadFile(dwdODServer, cosmoD2Path + "00/clc/", "cosmo-d2_germany_regular-lat-lon_model-level_2018111600_001_52_CLC.grib2.bz2", rawDataDir) 
    """
    fullUrl = serverName + path + fileName
    fullFile = targetDir + fileName
    print("Now attempting connection to {}".format(fullUrl))
    with urllib.request.urlopen(fullUrl) as response:
        with open(fullFile, "wb") as fileHandle:
            print("Now saving data in {}".format(fullFile))
            data = response.read()  # a bytes-object
            fileHandle.write(data)



def ftpDownloadFile(serverName, path, fileName, targetDir):
    """
    >>> ftpDownloadFile(dwdFtpServer, radolanPath, "RW-20180101.tar.gz", rawDataDir)
    """
    fullFile = targetDir + fileName
    print("Now attempting connection to {}".format(serverName))
    with FTP(serverName) as ftp:
        with open(fullFile, "wb") as fileHandle:
            ftp.login()
            print("Now moving to path {}".format(path))
            ftp.cwd(path)
            print("Now saving data in {}".format(fullFile))
            ftp.retrbinary("RETR " + fileName, fileHandle.write)



class MyFtpServer:
    """ macht dasselbe wie ftpDownloadFile, aber stateful, so dass nicht jedes mal
    neue Verbindung erzeugt wird."""

    def __init__(self, serverName, user=None, passwd=None, proxy=None):
        self.serverName = serverName
        self.user = user
        self.passwd = passwd
        self.proxy = proxy
        self.tryConnectNTimes(3)

    def tryConnectNTimes(self, n):
        try:
            self.connect(self.serverName, self.user, self.passwd, self.proxy)
        except EOFError as e:
            if n > 0:
                print("Connection error; retrying in a second ...")
                sleep(1)
                self.tryConnectNTimes(n-1)
            else: 
                raise e

    def connect(self, serverName, user=None, passwd=None, proxy=None):
        if not user:
            user = "anonymous"
        if not proxy:
            print("Now connecting to {}@{} using {}".format(user, serverName, passwd))
            self.server = FTP(serverName)
            self.server.login(user, passwd)
        else:
            userString = "{}@{}".format(user, serverName)
            print("Now connecting to {}@{} using {}".format(userString, proxy, passwd))
            self.server = FTP(proxy)
            self.server.login(userString, passwd)
        print("Connection established.")
        return True


    def __del__(self):
        print("Now deleting Ftp-Server")
        #if self.server:
        #    self.server.quit()

    def tryDownloadNTimes(self, path, fileName, targetDir, n):
        try:
            self.downloadFile(path, fileName, targetDir)
        except EOFError as e:
            if n > 0:
                print("Download error; retrying in a second ...")
                sleep(1)
                self.tryDownloadNTimes(path, fileName, targetDir, n-1)
            else:
                raise e

    def downloadFile(self, path, fileName, targetDir):
        fullFile = targetDir + fileName
        self.server.cwd("/")
        with open(fullFile, "wb") as fileHandle:
            print("Now moving to path {}".format(path))
            self.server.cwd(path)
            print("Now saving data in {}".format(fullFile))
            self.server.retrbinary("RETR " + fileName, fileHandle.write)
