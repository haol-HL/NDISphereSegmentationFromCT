import os

from hagFtpClient import *
from hagLogger import *
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
        ct_source_list = list(collection_ctsource.find({}))
        ct_source_name = ct_source_list[0]['data_name']
        if ct_source_name in ct_source_name_list:
            continue
        ftp_ctsource_file_name = ct_source_list[0]['file_path']

        ct_seg_list = list(collection_seg.find({}))
        ftp_ctseg_file_name = ct_seg_list[len(ct_seg_list)-1]['file_path']
        if len(ftp_ctseg_file_name) == 0:
            
            print(ct_source_name, 'has no result image!')
            continue
        ct_source_name_list.append(ct_source_name)
        ftp_file_name_list.append([ftp_ctsource_file_name, ftp_ctseg_file_name])
    
    return ct_source_name_list, ftp_file_name_list

if __name__ == '__main__':
    
    ftpath = ''  # 远程文件夹
    localpath = 'C:\luohao\SphereSegDATA/'  
    ctsource_names, ftp_files = GetCtSourceListFromMongo("192.168.0.100", 27017)
    ip = '192.168.0.100'
    port = '21'
    user = 'hanglok-ftp'
    password = 'hanglok'
    logger = HagLogger("log.log")
    ftp = HagFTPClient(ip, port, user, password, logger)
    
    print("{} unique ct source Imgae in total.".format(len(ctsource_names)))

    for name, ftpf in zip(ctsource_names, ftp_files):
        localCTpath = localpath + name + r'/'
        if not os.path.exists(localCTpath):
            os.makedirs(localCTpath)
        ftp.down_file(ftpath + ftpf[0], localCTpath + ftpf[0])
        ftp.down_file(ftpath + ftpf[1], localCTpath + ftpf[1])
        print(name, " DownLoaded")
    ftp.close()

    