# -*- coding: utf-8 -*-
# 机翻软件win11 上控制界面
import PySimpleGUI as sg
from tkinter.filedialog import askdirectory
import os
import subprocess
from tkinter import messagebox
import logging
logging.basicConfig(filename='logs.log', level=logging.DEBUG)
import shutil
import threading
from PIL import Image
import cv2
import numpy as np

# 原始图片为 jpg，输出图片为png
# 当前程序所在绝对路径
rootdir = os.getcwd()
# 执行脚本前半部分
pybin = os.path.join(rootdir,
                     'Scripts\python.exe -m manga_translator  --translator=none --mode batch --target-lang=CHS ')
# 使用gpu
usecuda = False
# 擦除进程pid
cmdpid = None
# 提取中文文字进程pid
chinesepid = None
# 进度 1=未开始，2=擦除进行中，3=提取中文进行中，4=全部结束
fugaiwenzi = 1
# 当前执行状态
current_status = 'no'

layout = [
    [sg.Text('待翻译图片文件夹', background_color="#e3f2fd", text_color='#212121'), sg.InputText(key="sourceinput"),
     sg.Button('选择文件夹', key="sourcedir", enable_events=True, button_color='#018fff', border_width=0)],
    [sg.Text('输出到目标文件夹', background_color="#e3f2fd", text_color='#212121'), sg.InputText(key="targetinput"),
     sg.Button('选择文件夹', key="targetdir", enable_events=True, button_color='#018fff', border_width=0)],

    [
        sg.Button('开始执行', key="startbtn", button_color='#2196f3', size=(16, 2), font=16),
        sg.Checkbox("使用GPU", default=True, key="usegpu", background_color="#e3f2fd", checkbox_color="#eeeeee",
                    text_color="#ff0000", size=(9, 1))
    ],
    [
        sg.Text('', key="logs", background_color="#e3f2fd", text_color='#212121')
    ]
]
sg.theme('Material1')

# 删除tmp下文件
def delete_tmp_files():
    # 指定tmp目录路径
    dir_path = os.path.join(rootdir, 'tmp')

    # 获取所有文件和目录的列表
    items = os.listdir(dir_path)

    # 遍历所有文件和目录
    for item in items:
        # 构造完整路径
        file_path = os.path.join(dir_path, item)

        # 判断是否为文件
        if os.path.isfile(file_path):
            # 删除文件
            os.remove(file_path)

        # 判断是否为目录
        if os.path.isdir(file_path):
            # 删除目录及子目录和文件
            shutil.rmtree(file_path)


def logto(msg):
    logging.debug(msg)

# 日文删除非气泡
def delnoqipao():
    # image1 是文字图 在 /tmp里
    # image2 是日文无字图，在目标的 riwen里
    obj = {
        # 原始日文图片，用于排除非气泡
        "source": os.path.join(window['sourceinput'].get(), 'riwen'),
        # 日文图片的掩码图
        "mask": os.path.join(rootdir, 'tmp'),
        # 已擦除气泡文字的日文图
        "target": window['targetinput'].get()
    }
    for name in os.listdir(obj['source']):
        # name 是原始图像名，中文 日文图像名一样
        # 读取需要提取文字的原始中文图像
        img = cv2.imread(os.path.join(obj['source'], name))

        # 读取中文掩码并反转
        # 中文掩码
        maskimg = os.path.join(obj['mask'], name + '.mask_final.png')
        
        # 读取掩码图像，掩码图在项目的result中会生成，可以在生成代码那里添加一些语句，让它每次多写一份到指定的地方
        mask = cv2.imread(maskimg, cv2.IMREAD_GRAYSCALE)

        # 确保两张图片大小相同
        #assert raw_mask.shape == img.shape[:2]

        # 二值化 raw_mask，255 表示白色，0 表示黑色
        _, binary_raw_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 对 raw_mask 进行连通域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_raw_mask, 8, cv2.CV_32S)

        # 从第1个连通域开始遍历，第0个连通域是背景
        for i in range(1, num_labels):
            # 获取当前连通域的信息
            x, y, w, h, size = stats[i]

            # 获取连通域对应的 img 的区域
            img_region = img[y:y+h, x:x+w]

            # 判断连通域周围是否有非白色点，这里判断的逻辑是只要 RGB 中有一个小于 250 的就认为不是白色
            #if np.any(img_region < 200):
            if np.count_nonzero(img_region == 0) > 0:
                # 将 raw_mask 中的这个连通域删除，即设置为黑色
                labels[labels == i] = 0

        # 更新 raw_mask
        mask[labels > 0] = 255
        mask[labels == 0] = 0
        
        cv2.imwrite(os.path.join(obj['target'],name).replace('.jpg','.png'), mask)




# 1. 首先从 sourceinput/zhongwen 中获取中文图像
# 2. 然后从 /tmp/中文名.mask_final.png 取出中文图像的掩码图，并反转
# 3. 将此反转图和 sourceinput/zhongwen/图像名 交集取得纯文字图
# 4. 将此纯文字图和 已生成的日文空白气泡图拼接， targetinput/riwen/图像名，得到最终图
# 5. 删除tmp图片，删除 targetinput/zhongwen
def darken_blend():
    global fugaiwenzi
    # image1 是文字图 在 /tmp里
    # image2 是日文无字图，在目标的 riwen里
    obj = {
        # 原始中文图片，需获取文字
        "source": os.path.join(window['sourceinput'].get(), 'zhongwen'),
        # 中文图片的掩码图
        "mask": os.path.join(os.getcwd(), 'tmp'),
        # 已擦除气泡文字的日文图
        "riwen": window['targetinput'].get()
    }
    for name in os.listdir(obj['source']):
        # name 是原始图像名，中文 日文图像名一样
        # 读取需要提取文字的原始中文图像
        img = cv2.imread(os.path.join(obj['source'], name))

        # 读取中文掩码并反转
        # 中文掩码
        maskimg = os.path.join(obj['mask'], name + '.mask_final.png')
        
        # 读取掩码图像，掩码图在项目的result中会生成，可以在生成代码那里添加一些语句，让它每次多写一份到指定的地方
        mask = cv2.imread(maskimg, cv2.IMREAD_GRAYSCALE)
        
        #========================
        
      

        #====================
        
        
        # 将掩膜逻辑反转
        inverse_mask = cv2.bitwise_not(mask)
        # 翻转后覆盖原图
        cv2.imwrite(maskimg, inverse_mask)

        ## 和原始中文图做交集，取出纯文字
        img_inpainted = np.copy(img)
        img_inpainted[inverse_mask > 0] = np.array([255, 255, 255], np.uint8)
        result = img_inpainted
        # 保存文字图 覆盖原图
        cv2.imwrite(maskimg, result)

        # 开始拼接 将两个图像加载为Pillow对象
        img1 = Image.open(maskimg)
        img2 = Image.open(os.path.join(obj['riwen'], name).replace('.jpg', '.png'))

        # 确保两个图像具有相同的大小
        img1 = img1.resize(img2.size)

        # 将两个图像转换为RGBA模式
        img1 = img1.convert("RGBA")
        img2 = img2.convert("RGBA")

        # 获取图像的像素数据
        data1 = img1.getdata()
        data2 = img2.getdata()

        # 用"darken"效果混合两个图像
        new_data = []
        for pixel1, pixel2 in zip(data1, data2):
            # 从每个像素中获取RGB值和Alpha通道值
            r1, g1, b1, a1 = pixel1
            r2, g2, b2, a2 = pixel2

            # 比较每个通道的值，选择较小的值作为新像素的值
            r = min(r1, r2)
            g = min(g1, g2)
            b = min(b1, b2)
            a = min(a1, a2)

            # 创建新像素并添加到新数据列表中
            new_pixel = (r, g, b, a)
            new_data.append(new_pixel)

        # 创建新图像 覆盖日文图像，结束
        result_image = Image.new("RGBA", img1.size)
        result_image.putdata(new_data)
        result_image.save(os.path.join(obj['riwen'], name).replace('.jpg', '.png'))
    fugaiwenzi = 4


# 开始擦除日文图片文字
def clearjaptext(obj):
    global cmdpid
    # 写入标志，进行中
    cmd = "{pybin} -v -i {sourcedir} -o {targetdir}   ".format(pybin=pybin, sourcedir=obj['sourcedir'],
                                                             targetdir=obj['targetdir'])
    if obj['usegpu']:
        cmd += " --use-cuda"
    """ stdout=subprocess.PIPE, stderr=subprocess.PIPE,  """
    delete_tmp_files()
    proc = subprocess.Popen(cmd, bufsize=0, shell=True)
    cmdpid = proc.pid
    print('开始执行【擦除日文图片文字】，cmd=%s' % cmd)



# 停止某个进程
def stopocr(pid):
    print("停止进程pid=【pid】")
    # 正常停止
    if pid and int(pid) > 0:
        print(os.system("tskill %s" % pid))


# 开始提取生成中文图片的掩码图 mask_final.png
def getchinese():
    global chinesepid,fugaiwenzi
    #delnoqipao()
    obj = {
        "sourcedir": os.path.join(window['sourceinput'].get(), 'zhongwen'),
        "targetdir": os.path.join(rootdir,'tmp'),
        "usegpu": window['usegpu'].get()
    }
    cmd = "{pybin}  -v -i {sourcedir} -o {targetdir}   ".format(pybin=pybin, sourcedir=obj['sourcedir'],
                                                                targetdir=obj['targetdir'])
    if obj['usegpu']:
        cmd += " --use-cuda"
    # stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    delete_tmp_files()
    proc = subprocess.Popen(cmd, bufsize=0, shell=True)
    chinesepid = proc.pid
    fugaiwenzi = 2  # 进入第二个阶段
    print('启动================================%s'%cmd)


# 判断进程是否存在
def pidlive(pid):
    if not pid or int(pid) < 1:
        return False
    # 执行tasklist命令，获取所有正在运行的进程列表
    os.system("tasklist /FI \"PID eq {}\" > process_list.txt".format(pid))

    # 读取任务管理器列表
    process_list = ""
    with open("process_list.txt", "r") as f:
        process_list = f.read()

    # 判断要查询的进程是否存在于任务管理器列表中
    if str(pid) in process_list:
        return True
    else:
        # print("进程不存在")
        return False


# 定义读取输出和错误输出的函数
def read_output(stream):
    while True:
        line = stream.readline()
        if not line:
            break
        print(line.decode(encoding='utf-8').strip())


window = sg.Window('Zero', layout, size=(1000, 400), icon=os.path.join(rootdir, "icon.ico"),resizable=True)

while True:
    event, values = window.read(timeout=100)
    # print('event',event)
    # print('values',values)
    # 选择源图片文件夹
    if event == 'sourcedir':
        window['sourceinput'].update(askdirectory())
    # 选择目标图片文件夹
    elif event == 'targetdir':
        window['targetinput'].update(askdirectory())
    elif event == 'startbtn':
        # 当前执行中，点击按钮关闭
        if current_status == 'ing':
            fugaiwenzi = 1
            current_status = "no"
            window['startbtn'].update(text="开始执行")
            stopocr(cmdpid)
            stopocr(chinesepid)
        else:
            current_status = "ing"
            # 开始执行 日文擦除
            obj = {
                # 日文图片目录
                "sourcedir": window['sourceinput'].get(),
                "targetdir": '',
                "usegpu": window['usegpu'].get()
            }
            if not obj['sourcedir']:
                messagebox.showerror('出错了', '必须选择要翻译的图片文件夹', parent=window.TKroot)
                current_status = "no"
                continue

            if not os.path.exists(obj['sourcedir']):
                messagebox.showerror('出错了', '你选择的文件夹不存在')
                current_status = "no"
                continue
            if not os.path.exists(os.path.join(obj['sourcedir'], 'riwen')):
                messagebox.showerror('出错了', '你选择的文件夹中不存在riwen目录')
                current_status = "no"
                continue

            if not obj['targetdir']:
                # 存到目录的riwen子文件夹下
                obj['targetdir'] = window['sourceinput'].get() + "-translated"
            window['targetinput'].update(value=obj['targetdir'])

            obj['sourcedir']=os.path.join(obj['sourcedir'],'riwen')
            obj['targetdir']=obj['targetdir']
            # obj['targetdir']=os.path.join(obj['targetdir'],'riwen')

            window['startbtn'].update(text="执行中,点击停止")
            window['logs'].update(value="执行日语漫画文字擦除中...\n")
            print("开始执行")
            #  
            # fugaiwenzi=2
            clearjaptext(obj)

    elif event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
        current_status = 'no'
        stopocr(cmdpid)
        stopocr(chinesepid)
        break
    elif current_status == 'ing':
        # 擦除进程如果仍在执行，则继续
        if pidlive(cmdpid):
            continue
        # 擦除进程已结束

        # 如果fugaiwenzi=1。尚未生成掩码，开始执行生成掩码操作
        if fugaiwenzi == 1:

            getchinese()
            continue
        # 在执行生成掩码操作
        if fugaiwenzi == 2:
            window['logs'].update(value="进行中文气泡文字提取中...")
            # 判断生成掩码图是否结束
            if pidlive(chinesepid):
                continue
            # 已结束，则启动合并
            fugaiwenzi = 3
            # 启动 合并
            threading.Thread(target=darken_blend).start()
            continue
        # 合并中
        if fugaiwenzi == 3:
            window['logs'].update(value="合并进行中")
            continue
        # 全部执行结束
        if fugaiwenzi == 4:
            current_status = 'no'
            window['logs'].update(value="全部结束")
            window['startbtn'].update(text="开始执行")

window.close()
