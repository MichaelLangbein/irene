import bz2
import tarfile
import datetime as dt
import urllib.request
from ftplib import FTP
import ftplib


def extract(path, fileName):
    fullName = path + fileName
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

    def __init__(self, serverName):
        self.server = FTP(serverName)
        self.server.login()

    def __del__(self):
        self.server.quit()

    def downloadFile(self, path, fileName, targetDir):
        fullFile = targetDir + fileName
        self.server.cwd("/")
        with open(fullFile, "wb") as fileHandle:
            print("Now moving to path {}".format(path))
            self.server.cwd(path)
            print("Now saving data in {}".format(fullFile))
            self.server.retrbinary("RETR " + fileName, fileHandle.write)
