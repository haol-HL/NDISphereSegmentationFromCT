# -*- encoding=utf-8 -*-
import ftplib
import os
import datetime
from hagLogger import HagLogger


class HagFTPClient:
    def __init__(self, ip, port, user, password, logger):
        self.ip = ip
        self.port = int(port)
        self.user = user
        self.password = password
        self.ftp = ftplib.FTP()
        self.ftp.set_pasv(False)
        self.logger = logger

        connect_ret = self.connect()
        if connect_ret:
            self.login()
            print("ftp login finish")

    def connect(self):
        ret = False
        try:
            self.ftp.connect(self.ip, self.port)  # 连接ftp
            self.logger.info('Connection ftp success')
            ret = True
        except Exception as e:
            self.logger.error('Connection ftp fail:{}'.format(e))

        return ret

    def login(self):
        ret = False
        try:
            self.ftp.login(self.user, self.password)  # 登录ftp
            self.logger.info('Login ftp success')
            ret = True
        except Exception as e:
            self.logger.error('Login ftp fail:{}'.format(e))
        return ret

    def close(self):
        try:
            self.ftp.close()  # 关闭
            self.logger.info('Close ftp success')
        except Exception as e:
            self.logger.error('Close ftp fail:{}'.format(e))

    def down_file(self, ftp_file, save_local_path):
        success = False
        abs_path = os.path.abspath(save_local_path)
        try:
            with open(abs_path, 'wb') as f:
                down_start = datetime.datetime.now()
                ret = self.ftp.retrbinary('RETR ' + ftp_file, f.write)  # 下载文件
                down_stop = datetime.datetime.now()
                self.logger.info("download cost time(m): {}".format((down_stop-down_start).seconds*1000))
                self.logger.info('Down ftp file return:{}'.format(ret))
                if ret.startswith('226'):
                    self.logger.info('Down ftp file success, save to:{}'.format(abs_path))
                    success = True
        except Exception as e:
            self.logger.error('Down ftp file fail:{}'.format(e))
            self.logger.error('Fail path:{}'.format(ftp_file))
            success = False
        return success

    def upload_file(self, ftp_path, local_path):
        success = False
        abs_path = os.path.abspath(local_path)
        try:
            with open(abs_path, 'rb') as f:
                up_start = datetime.datetime.now()
                ret = self.ftp.storbinary('STOR ' + ftp_path, f, 1024*1024)  # 上传文件
                up_stop = datetime.datetime.now()
                self.logger.info("upload cost time(m): {}".format((up_stop-up_start).seconds*1000))
                self.logger.info('Upload ftp file return:{}'.format(ret))
                if ret.startswith('226'):
                    self.logger.info('Upload ftp file success, save to:{}'.format(ftp_path))
                    success = True
        except Exception as e:
            self.logger.error('upload ftp file fail:{}'.format(e))
            self.logger.error('Fail path:{}'.format(abs_path))
            success = False
        return success


def debug():
    ip = '192.168.0.100'
    port = '21'
    user = 'hanglok-ftp'
    password = 'hanglok'
    logger = HagLogger("log.log")
    ftp = HagFTPClient(ip, port, user, password, logger)
    # ftp.down_file('00bb62b8-7dee-4e50-9d61-7be1e532d4f0[2021-06-02_15-13-31[MRI[_C1424951010115_8.mha',
    #               'C1424951010115_8.mha')
    # ftp.upload_file("remote_log.log", "local_log.log")
    ftp.close()
