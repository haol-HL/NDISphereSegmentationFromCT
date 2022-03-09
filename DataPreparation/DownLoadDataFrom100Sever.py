import os
import sys
import FTPdownloader
from pymongo import MongoClient

def GetCtSourceListFromMongo(ip:str, port:int):
    ct_source_name_list = []
    ftp_file_name_list = []
    client=MongoClient(ip,port)  
    for db_name in client.list_database_names():
        if len(db_name)<35:
            continue
        db = client[db_name]
        collection_ctsource = db['ct_source']
        collection_seg = db['ct_result']
        if (len(list(collection_ctsource.find({})))) == 0 or (len(list(collection_seg.find({}))) == 0):
            continue
        ct_source_name = list(collection_ctsource.find({}))[0]['data_name']
        ftp_ctsource_file_name = list(collection_ctsource.find({}))[0]['file_path']
        ftp_ctseg_file_name = list(collection_seg.find({}))[0]['file_path']
        if ct_source_name in ct_source_name_list:
            continue
        ct_source_name_list.append(ct_source_name)
        ftp_file_name_list.append([ftp_ctsource_file_name, ftp_ctseg_file_name])
    
    return ct_source_name_list, ftp_file_name_list

if __name__ == '__main__':
    # 输入参数
    ftpserver = '192.168.0.100' # ftp主机IP
    port = 21                                  # ftp端口
    usrname = 'hanglok-ftp'       # 登陆用户名
    pwd = 'hanglok'       # 登陆密码
    ftpath = '/home/hanglok-ftp/'  # 远程文件夹
    localpath = 'C:\luohao\SphereSegDATA/'  

    ctsource_names, ftp_files = GetCtSourceListFromMongo("192.168.0.100", 27017)
    
    Ftp = FTPdownloader.FtpDownloadCls(ftpserver, port, usrname, pwd)
    print("{} unique ct source Imgae in total.".format(len(ctsource_names)))

    for name, ftpf in zip(ctsource_names, ftp_files):
        localCTpath = localpath + name + r'/'
        if not os.path.exists(localCTpath):
            os.makedirs(localCTpath)
        Ftp.downloadFile(ftpath + ftpf[0], localCTpath + ftpf[0])
        Ftp.downloadFile(ftpath + ftpf[1], localCTpath + ftpf[1])
        print(name, " DownLoaded")
    Ftp.ftpDisConnect()

    