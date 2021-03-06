import os


def gen_train_txt(txt_path, img_dir):
    f = open(txt_path, 'w')
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)  # 获取各类的文件夹 绝对路径
            img_list = os.listdir(i_dir)  # 获取类别文件夹下所有png图片的路径
            for i in range(len(img_list)):
                if not img_list[i].endswith('png'):  # 若不是png文件，跳过
                    continue
                label_path = os.path.join("../LOLdataset/our485/label/", img_list[i])

                img_path = os.path.join("../LOLdataset/our485/img/", img_list[i])
                line = img_path + '***' + label_path + '\n'
                print(line)
                f.write(line)
            break
    f.close()


def gen_test_txt(txt_path, img_dir):
    f = open(txt_path, 'w')
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)  # 获取各类的文件夹 绝对路径
            img_list = os.listdir(i_dir)  # 获取类别文件夹下所有png图片的路径
            for i in range(len(img_list)):
                if not img_list[i].endswith('png'):  # 若不是png文件，跳过
                    continue
                label_path = os.path.join("./LOLdataset/eval15/high/", img_list[i])
                img_path = os.path.join("../LOLdataset/eval15/low/", img_list[i])
                line = img_path + '***' + label_path + '\n'
                f.write(line)
            break
    f.close()


def gen_result_txt(txt_path, img_dir):
    f = open(txt_path, 'w')
    res_list = os.listdir(img_dir)
    res_list.sort(key=lambda x: int(x[:-4]))
    label_list = os.listdir("../fivek_dataset/expert")
    # label_list.sort(key=lambda x: int(x[:-4]))
    for i in range(len(res_list)):
        res_path = os.path.join("result", res_list[i])
        label_path = os.path.join("../fivek_dataset/expert", label_list[i])
        line = res_path + "***" + label_path
        f.write(line + "\n")
    f.close()


if __name__ == '__main__':
    # gen_train_txt("./lol_train.txt", "../LOLdataset/our485")
    gen_result_txt("pre_result.txt", "./result")
