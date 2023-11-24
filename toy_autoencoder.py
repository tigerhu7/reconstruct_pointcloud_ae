######################################################################################
# Modified AutoEncoder by Hufeng

__author__ = 'Hufeng'

# base on sskqgfnnh
# base on the demo of GuoYang
# Check
#####################################################################################


# LIBRARIES
#----------------------------------------------------------------------------------
import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from numpy import genfromtxt
import ssk

np.set_printoptions(threshold=None)
#np.set_printoptions(threshold=np.nan)


# m_gpu = torch.cuda.device_count()-1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#torch.cuda.set_device(0)
print( 'Is Cuda Available?? ', torch.cuda.is_available() )
print( 'Torch Version?? ', torch.__version__ )



# CONFIGURATION
#-------------------------------------------------------------------------------------
model_name = 'Arch_Spiral'

train_model, save_checkpoint = False, False    # 这个True 的话， 就会把模型改了~~~~
#train_model, save_checkpoint = True, True


get_result = True
get_meshlab_result = True


print('*** Model Name *** ', model_name)
print('Train Model ?? ', train_model)
print('Save Model Checkpoint ?? ', save_checkpoint)
print('Get Result ?? ', get_result)
print('Meshlab Result ?? ', get_meshlab_result)


import os
# Get the directory of the current file
current_directory = os.path.dirname(os.path.abspath(__file__))
print("Directory of the current file:", current_directory)


work_dir = current_directory
data_dir = work_dir+'/DataSamples/'
model_dir = work_dir+'/Models/'

result_dir = work_dir + '/Results/test_231122'
os.makedirs( result_dir, exist_ok=True)

writer = SummaryWriter( work_dir+'/logs')

# GROUND TRUTH
#-------------------------------------------------------------------------------------
# Archimedean spiral
# rho(theta) = (a+b*theta)*( cos(w*theta), sin(w*theta) )
xx = 9

a = 0.01
b = 0.06
w = 5
#T = 1.618
T = 1.1
#T = 0.8
#T = 0.3
theta_series = np.arange(0, T, 0.001)


# Input Manifold
# 这些点可以当做是 ground truth点 ~~~
rho_x = (a+b*theta_series)*np.cos(w*theta_series)  
rho_y = (a+b*theta_series)*np.sin(w*theta_series)

dataset = np.vstack((rho_x, rho_y)).T


#plt.plot(rho_y, rho_x)
#plt.show()


# Latent Representation
phi_theta = theta_series


np.savetxt( data_dir+'/arch_spiral.txt', dataset, delimiter=';' )
#np.savetxt( data_dir+'/arch_spiral_meshlab.txt', np.column_stack((dataset, np.zeros(dataset.shape[0]))), delimiter=';' )
np.savetxt( data_dir+'/meshlab_input_arch_spiral_3D.txt', np.column_stack((dataset, np.zeros(dataset.shape[0]))), delimiter=';' )



# MODEL
#-------------------------------------------------------------------------------------
class autoencoder(nn.Module):
    
    # 每个 layers 定义好
    def __init__(self):

        super(autoencoder, self).__init__()
        self.layer1 = nn.Linear(2, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64  , 32  )
        self.layer4 = nn.Linear(32  , 16  )
        self.layer5 = nn.Linear(16  , 8)
        self.layer6 = nn.Linear(8, 1)

        self.layer7 = nn.Linear(1, 8)
        self.layer8 = nn.Linear(8, 16  )
        self.layer9 = nn.Linear(16  , 32  )
        self.layer10 = nn.Linear(32  , 64  )
        self.layer11 = nn.Linear(64  , 128  )
        self.layer12 = nn.Linear(128  , 2)

    # 然后 这些 layers 按照一定的顺序 排列好～～～～
    def forward(self, x):
        s = []
        neural_count = 0
        x = self.layer1(x)
        s1 = torch.sign(x)    # 是不是被激活了～～
        s1 = s1.data.cpu().numpy() + 1
        #s1 = s1.astype(np.bool)
        s1 = s1.astype(bool)
        
        neural_count = neural_count + s1.shape[1]
        s.append(s1)
        x = F.relu(x)

        x = self.layer2(x)
        s2 = torch.sign(x)
        s2 = s2.data.cpu().numpy() + 1
        s2 = s2.astype(bool)
        neural_count = neural_count + s2.shape[1]
        s.append(s2)
        x = F.relu(x)

        x = self.layer3(x)
        s3 = torch.sign(x)
        s3 = s3.data.cpu().numpy() + 1
        s3 = s3.astype(bool)
        neural_count = neural_count + s3.shape[1]
        s.append(s3)
        x = F.relu(x)

        x = self.layer4(x)
        s4 = torch.sign(x)
        s4 = s4.data.cpu().numpy() + 1
        s4 = s4.astype(bool)
        neural_count = neural_count + s4.shape[1]
        s.append(s4)
        x = F.relu(x)

        x = self.layer5(x)
        s5 = torch.sign(x)
        s5 = s5.data.cpu().numpy() + 1
        s5 = s5.astype(bool)
        neural_count = neural_count + s5.shape[1]
        s.append(s5)
        x = F.relu(x)

        x = self.layer6(x)
        z = x

        x = self.layer7(x)
        s7 = torch.sign(x)
        s7 = s7.data.cpu().numpy() + 1
        s7 = s7.astype(bool)
        neural_count = neural_count + s7.shape[1]
        s.append(s7)
        x = F.relu(x)

        x = self.layer8(x)
        s8 = torch.sign(x)
        s8 = s8.data.cpu().numpy() + 1
        s8 = s8.astype(bool)
        neural_count = neural_count + s8.shape[1]
        s.append(s8)
        x = F.relu(x)

        x = self.layer9(x)
        s9 = torch.sign(x)
        s9 = s9.data.cpu().numpy() + 1
        s9 = s9.astype(bool)
        neural_count = neural_count + s9.shape[1]
        s.append(s9)
        x = F.relu(x)

        x = self.layer10(x)
        s10 = torch.sign(x)
        s10 = s10.data.cpu().numpy() + 1
        s10 = s10.astype(bool)
        neural_count = neural_count + s10.shape[1]
        s.append(s10)
        x = F.relu(x)

        x = self.layer11(x)
        s11 = torch.sign(x)
        s11 = s11.data.cpu().numpy() + 1
        s11 = s11.astype(bool)
        neural_count = neural_count + s11.shape[1]
        s.append(s11)
        x = F.relu(x)

        x = self.layer12(x)

        xsize = x.shape[0]
        ss = np.zeros(shape=[xsize, neural_count], dtype=bool)  # 多少个神经节点~~
        c1 = 0
        for i in range(len(s)):
            c0 = 0 if i == 0 else c1
            c1 = s[i].shape[1] + c0
            ss[:, c0: c1] = s[i]
        return x, z, ss   # predict data， latent space， activated neurons~~


# Supplimentaries
#-------------------------------------------------------------------------------------
# loss function 是传统的 MSE
def test_loss_fun(x, y):
    z = x-y
    z = z * z
    z = np.sum(z, axis=1)
    z = np.sqrt(z)
    return z.mean()


def train_loop(in_dataset):

    #【1】 Hyper Parameters  超参数
    # num_epochs = 8000
    # batch_size = 258
    # learning_rate = 8.0e-5


    num_epochs = 20000
    batch_size = 230
    learning_rate = 9.0e-5

    # num_epochs = 7000
    # batch_size = 300
    # learning_rate = 1.0e-5

    dataset = in_dataset+0

    # 【2】 Choose Dataset
    #dataset = genfromtxt(model_name + '.csv', delimiter=',')
    np.random.shuffle(dataset)
    dataset_size = dataset.shape[0]
    train_size = dataset_size
    dataset_tensor = torch.from_numpy(dataset).float()
    dataloader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)

    # 【3】 Model and Optimization
    model = autoencoder().cuda()   # 模型使用GPU
    criterion = nn.MSELoss()  # loss 使用 MSE
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0)  # 优化方法， 使用 Adam

    # 【2.5】 Pick Testing Data from Dataset
    """prepare test data"""
    testdata_tensor = torch.from_numpy(dataset).float()   # numpy 的值转变为 tensor
    numX = dataset_size  # - train_size
    testloader = DataLoader(testdata_tensor, batch_size=numX, shuffle=False)
    
    # 【4】 Train and Test Loss
    # 应该是训练， 然后使用 test data 算一算loss， 每一个 epoch 做这么一次~~~
    # 每一个 epoch， 用 test data 算一下loss，
    # 如果 loss 减少了， 就更新模型的checkpoint

    min_tot_loss = 1e99
    """training"""
    for epoch in range(num_epochs):
        for data in dataloader:
            img = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            # ===================forward=====================
            y, z, _ = model(img)
            loss = criterion(y, img)
            # ===================backward====================a
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """test"""
        with torch.no_grad():
            for data in testloader:
                point = data
                point = point.view(point.size(0), -1)
                point = Variable(point).cuda()
                point_new, _, _ = model(point)
                # test_loss0 = criterion(point, point_new)
                test_loss = test_loss_fun(point.data.cpu().numpy(), point_new.data.cpu().numpy())
                #test_loss = ssk.test_loss_fun(point.data.cpu().numpy(), point_new.data.cpu().numpy())
                # ===================log========================

                
                if loss.item() < min_tot_loss:
                    min_tot_loss = loss.item()
                    print('epoch [{}/{}], loss:{:.4e}, test_loss:{:.4e} *'
                        .format(epoch + 1, num_epochs, loss.item(), test_loss))
                    
                    
                    writer.add_scalar("train_loss", loss.item(), epoch)
                    writer.add_scalar('test_loss', test_loss, epoch)

                    if save_checkpoint:   # 已经训练好的， 就不要再更新了， 
                        torch.save(model, model_dir + model_name + '_autoencoder.pth')
                else:
                    print('epoch [{}/{}], loss:{:.4e}, test_loss:{:.4e}'
                        .format(epoch + 1, num_epochs, loss.item(), test_loss))


    return model


def test_loop(testdata, model):


    # 对数据进行增强， batch归类， 等等， 用于训练或者测试的操作~~~~
    numX = testdata.shape[0]
    bsize = numX # batch size
    numX = numX - numX % bsize
    print('data num = {}'.format(numX))
    testdata_tensor = torch.from_numpy(testdata[0:numX, :]).float()
    testloader = DataLoader(testdata_tensor, batch_size=bsize, shuffle=False)
    

    output_list = []
    neural_list = []
    latent_list = []
    for data in testloader:
        
        # 初始数据
        x = data
        x = x.view(x.size(0), -1)
        x = Variable(x).cuda()
        
        # 模型推理
        y, z, s = model(x)   # 输入已知的model， 进行 inference 推理
        x = data.numpy()
        y = y.data.cpu().numpy()
        y = np.reshape(y, (bsize, -1))
        z = z.data.cpu().numpy()
        z = np.reshape(z, (bsize, -1))
        
        # 数据输出
        output_list.append(y)  # 进行 decode 之后的 data
        latent_list.append(z)  # latent space 的数据
        neural_list.append(s)  # 激活的神经元的数据


    # 结果转换为 可以展示的形式 
    neural_data = np.zeros(shape=[numX, neural_list[0].shape[1]])
    index = 0
    for neural in neural_list:
        s_shape = neural.shape
        neural_data[index: index + s_shape[0], :] = neural
        index = index + s_shape[0]
    index = 0
    latent_data = np.zeros(shape=[numX, dataset.shape[1]-1 ])
    for ldata in latent_list:
        latent_data[index: index + ldata.shape[0], :] = ldata
        index = index + ldata.shape[0]
    index = 0
    output_data = np.zeros(shape=[numX, dataset.shape[1]])
    for odata in output_list:
        output_data[index: index + odata.shape[0], :] = odata
        index = index + odata.shape[0]
    
    return output_data, latent_data, neural_data



# MAIN 
#---------------------------------------------------------------------------------------
if __name__ == '__main__':

    # TRAIN  训练模型~
    #-----------------------------------------------------------------------------------
    if train_model == True:
        
        if 1:
            model = train_loop(dataset)

        if 0:

            # 【1】 Hyper Parameters  超参数
            # num_epochs = 5000
            # batch_size = 258
            # learning_rate = 8.0e-5

            num_epochs = 7000
            batch_size = 300
            learning_rate = 1.0e-5


            # 【2】 Choose Dataset
            #dataset = genfromtxt(model_name + '.csv', delimiter=',')
            np.random.shuffle(dataset)
            dataset_size = dataset.shape[0]
            train_size = dataset_size
            dataset_tensor = torch.from_numpy(dataset).float()
            dataloader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)

            # 【3】 Model and Optimization
            model = autoencoder().cuda()   # 模型使用GPU
            criterion = nn.MSELoss()  # loss 使用 MSE
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=0)  # 优化方法， 使用 Adam

            # 【2.5】 Pick Testing Data from Dataset
            """prepare test data"""
            testdata_tensor = torch.from_numpy(dataset).float()   # numpy 的值转变为 tensor
            numX = dataset_size  # - train_size
            testloader = DataLoader(testdata_tensor, batch_size=numX, shuffle=False)
            
            # 【4】 Train and Test Loss
            # 应该是训练， 然后使用 test data 算一算loss， 每一个 epoch 做这么一次~~~
            # 每一个 epoch， 用 test data 算一下loss，
            # 如果 loss 减少了， 就更新模型的checkpoint

            min_tot_loss = 1e99
            """training"""
            for epoch in range(num_epochs):
                for data in dataloader:
                    img = data
                    img = img.view(img.size(0), -1)
                    img = Variable(img).cuda()
                    # ===================forward=====================
                    y, z, _ = model(img)
                    loss = criterion(y, img)
                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                """test"""
                with torch.no_grad():
                    for data in testloader:
                        point = data
                        point = point.view(point.size(0), -1)
                        point = Variable(point).cuda()
                        point_new, _, _ = model(point)
                        # test_loss0 = criterion(point, point_new)
                        test_loss = test_loss_fun(point.data.cpu().numpy(), point_new.data.cpu().numpy())
                        #test_loss = ssk.test_loss_fun(point.data.cpu().numpy(), point_new.data.cpu().numpy())
                        # ===================log========================

                        
                        if loss.item() < min_tot_loss:
                            min_tot_loss = loss.item()
                            print('epoch [{}/{}], loss:{:.4e}, test_loss:{:.4e} *'
                                .format(epoch + 1, num_epochs, loss.item(), test_loss))
                            
                            
                            writer.add_scalar("train_loss", loss.item(), epoch)
                            writer.add_scalar('test_loss', test_loss, epoch)

                            if save_checkpoint:   # 已经训练好的， 就不要再更新了， 
                                torch.save(model, model_dir + model_name + '_autoencoder.pth')
                        else:
                            print('epoch [{}/{}], loss:{:.4e}, test_loss:{:.4e}'
                                .format(epoch + 1, num_epochs, loss.item(), test_loss))


    else:
        model = torch.load( model_dir + model_name + '_autoencoder.pth')


    # GENERATE  训练好了之后， 保存加上颜色的结果，
    # 数据格式为
    #
    # in ambient space [ x, y, z, colorID, R, G, B ] 
    # 或者
    # in latent space [ x, y, colorID, R, G, B]
    #-----------------------------------------------------------------------------------
    if get_result == True:
        # record activation    # 把这些 activated 激活了的部分， 记录下来 ~~~~~~
        
        # 纯原始数据
        #testdata = genfromtxt(model_name + '.csv', delimiter=',')
        # 使用输入的dataset
        if 1:
            testdata = dataset[0:-1:3] + 0   # 挑选一些点
            test_theta_series = theta_series[0:-1:3] + 0

            if 1:
                output_data, latent_data, neural_data = test_loop(testdata, model)

            if 0:
                # 对数据进行增强， batch归类， 等等， 用于训练或者测试的操作~~~~
                numX = testdata.shape[0]
                bsize = numX # batch size
                numX = numX - numX % bsize
                print('data num = {}'.format(numX))
                testdata_tensor = torch.from_numpy(testdata[0:numX, :]).float()
                testloader = DataLoader(testdata_tensor, batch_size=bsize, shuffle=False)
                

                output_list = []
                neural_list = []
                latent_list = []
                for data in testloader:
                    
                    # 初始数据
                    x = data
                    x = x.view(x.size(0), -1)
                    x = Variable(x).cuda()
                    
                    # 模型推理
                    y, z, s = model(x)   # 输入已知的model， 进行 inference 推理
                    x = data.numpy()
                    y = y.data.cpu().numpy()
                    y = np.reshape(y, (bsize, -1))
                    z = z.data.cpu().numpy()
                    z = np.reshape(z, (bsize, -1))
                    
                    # 数据输出
                    output_list.append(y)  # 进行 decode 之后的 data
                    latent_list.append(z)  # latent space 的数据
                    neural_list.append(s)  # 激活的神经元的数据


                # 结果转换为 可以展示的形式 
                neural_data = np.zeros(shape=[numX, neural_list[0].shape[1]])
                index = 0
                for neural in neural_list:
                    s_shape = neural.shape
                    neural_data[index: index + s_shape[0], :] = neural
                    index = index + s_shape[0]
                index = 0
                latent_data = np.zeros(shape=[numX, dataset.shape[1]-1 ])
                for ldata in latent_list:
                    latent_data[index: index + ldata.shape[0], :] = ldata
                    index = index + ldata.shape[0]
                index = 0
                output_data = np.zeros(shape=[numX, dataset.shape[1]])
                for odata in output_list:
                    output_data[index: index + odata.shape[0], :] = odata
                    index = index + odata.shape[0]
                

            # 这些点， 再加上一些颜色， 然后保存下来~~~
            input_with_color = ssk.PlotData(testdata, neural_data, plot_type=0, savename = result_dir+'/test_neural.txt')
            output_with_color = ssk.PlotData(output_data, neural_data, plot_type=0, savename = result_dir+'/output_neural.txt')
            latent_with_color = ssk.PlotData(latent_data, neural_data, plot_type=0, savename = result_dir + '/latent_neural.txt')


        # 或者使用 整幅图, 用来显示ambient space partition~~~~~~
        if 1:  
            row_max = np.max(dataset[:,0])
            row_min = np.min(dataset[:,0])
            col_max = np.max(dataset[:,1])
            col_min = np.min(dataset[:,1])

            x = np.linspace(row_min, row_max, 100)
            y = np.linspace(col_min, col_max, 100)            
            X, Y = np.meshgrid(x, y)

            testdata_fullplane = np.vstack(( X.flatten(), Y.flatten() )).T
            output_data_fullplane, latent_data_fullplane, neural_data_fullplane = test_loop(testdata_fullplane, model)

            # 这些点， 再加上一些颜色， 然后保存下来~~~
            input_fullplane_with_color = ssk.PlotData(testdata_fullplane, neural_data_fullplane, plot_type = 0, savename = result_dir+'/test_neural_fullplane.txt')
            # output_fullplane_with_color = ssk.PlotData(output_data_fullplane, neural_data_fullplane, plot_type=0, savename = result_dir+'/output_neural_fullplane.txt')
            # latent_fullplane_with_color = ssk.PlotData(latent_data_fullplane, neural_data_fullplane, plot_type=0, savename = result_dir + '/latent_neural_fullplane.txt')
            
            input_fullplane_with_color_2 = ssk.PlotData(testdata_fullplane, neural_data_fullplane, plot_type = 2, savename = result_dir+'/test_neural_fullplane_2.txt')
   

        # 直接做显示  2*3 的 subplots
        if 1:
         
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')

            #fig, axs = plt.subplots(2,3, figsize = (12, 8))
            fig, ([ax1, ax2, ax3], [ax4, ax5, ax6])= plt.subplots(2,3, figsize = (12, 8))

            # ax.set_xlim(  np.min([row_min, col_min]), np.max([row_max, col_max]) )
            # ax.set_ylim(  np.min([row_min, col_min]), np.max([row_max, col_max]) )
            # ax.set_zlim(  -0.02, 0.02)
            # ax.set_aspect('equal')

            color_list = plt.cm.rainbow(np.linspace(0, 1, testdata.shape[0]))  # Generate rainbow colors


            #color_list = [ this_input[3:6]/255 for this_input in input_with_color ]
            ax1.scatter( input_with_color[:,0],  input_with_color[:,1], c = color_list, s = 5 ) # You can customize color ('c') and marker style ('marker')
            ax1.set_xlim(  row_min, row_max )
            ax1.set_ylim(  col_min, col_max )
            ax1.set_title( 'input manifold' )


            #color_list = [ this_input[2:5]/255 for this_input in latent_with_color ]
            ax2.scatter(  test_theta_series,  0*np.arange(latent_with_color[:,0].shape[0]), c = color_list, s = 3 )
            #ax2.set_xlim(  row_min, row_max )
            #ax2.set_ylim(  col_min, col_max )
            ax2.set_title( 'latent representation' )


            #color_list = [ this_input[3:6]/255 for this_input in output_with_color ]
            ax3.scatter( output_with_color[:,0],  output_with_color[:,1], c = color_list, marker='*', s = 3 ) # You can customize color ('c') and marker style ('marker')
            ax3.set_xlim(  row_min, row_max )
            ax3.set_ylim(  col_min, col_max )
            ax3.set_title( 'reconstructed manifold' )


            color_list = [ this_input[3:6]/255 for this_input in input_fullplane_with_color ]
            ax4.scatter( input_fullplane_with_color[:,0],  input_fullplane_with_color[:,1], c = color_list, s = 1 ) # You can customize color ('c') and marker style ('marker')
            ax4.scatter( input_with_color[:,0],  input_with_color[:,1], c = 'k', s = 1 ) # You can customize color ('c') and marker style ('marker')
            ax4.set_title( 'encoder decomposition' )


            color_list = [ this_input[3:6]/255 for this_input in input_fullplane_with_color_2 ]
            ax5.scatter( input_fullplane_with_color_2[:,0],  input_fullplane_with_color_2[:,1], c = color_list, s = 1 ) # You can customize color ('c') and marker style ('marker')
            ax5.scatter( input_with_color[:,0],  input_with_color[:,1], c = 'k', s = 1 ) # You can customize color ('c') and marker style ('marker')
            ax5.set_title( 'encoder_and_decoder decomposition' )

            plt.show()



            # # Set labels and title
            # ax.set_xlabel('X-axis')
            # ax.set_ylabel('Y-axis')
            # ax.set_title('3D Point Cloud Visualization')

            # color_list = [ this_input[2:5]/255 for this_input in output_with_color ]
            # ax.scatter( latent_with_color[:,0],  np.arange( latent_with_color.shape[0] ),  c = color_list, marker='*' ) # You can customize color ('c') and marker style ('marker')


            # Show the plot



        # 直接做显示   简单的~~
        if 0:
         
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.set_xlim(  np.min([row_min, col_min]), np.max([row_max, col_max]) )
            ax.set_ylim(  np.min([row_min, col_min]), np.max([row_max, col_max]) )

            ax.set_zlim(  -0.02, 0.02)

            ax.set_aspect('equal')

            color_list = [ this_input[3:6]/255 for this_input in input_with_color ]
            ax.scatter( input_with_color[:,0],  input_with_color[:,1], c = color_list, s = 5 ) # You can customize color ('c') and marker style ('marker')

            color_list = [ this_input[3:6]/255 for this_input in output_with_color ]
            ax.scatter( output_with_color[:,0],  output_with_color[:,1], c = color_list, marker='*', s = 3 ) # You can customize color ('c') and marker style ('marker')

            color_list = [ this_input[3:6]/255 for this_input in input_fullplane_with_color ]
            #ax.scatter( input_fullplane_with_color[:,0],  input_fullplane_with_color[:,1], c = color_list, s = 1 ) # You can customize color ('c') and marker style ('marker')

            # Set labels and title
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_title('3D Point Cloud Visualization')

            # color_list = [ this_input[2:5]/255 for this_input in output_with_color ]
            # ax.scatter( latent_with_color[:,0],  np.arange( latent_with_color.shape[0] ),  c = color_list, marker='*' ) # You can customize color ('c') and marker style ('marker')


            # Show the plot
            plt.show()



        # 把结果 保存成文件， 然后在 meshlab 或者其他软件中打开查看
        if 0:


            # # 保存这些点？？？
            # np.savetxt( result_dir + '/output_' + model_name + '.csv', output_data, delimiter=',')
            # np.savetxt( result_dir + '/latent_' + model_name + '.csv', latent_data, delimiter=',')
            # np.savetxt( result_dir + '/neural_' + model_name + '.csv', neural_data, delimiter=',')


            # np.savetxt( result_dir + '/input_arch_spiral.txt', testdata, delimiter=',')
            # np.savetxt( result_dir+'/input_arch_spiral_meshlab.txt', np.column_stack(( testdata, np.zeros(testdata.shape[0]))), delimiter=';')

            # np.savetxt( result_dir + '/output_arch_spiral.txt', output_data, delimiter=',')
            # np.savetxt( result_dir+'/output_arch_spiral_meshlab.txt', np.column_stack((output_data, np.zeros(output_data.shape[0]))), delimiter=';')


            # 输出为 meshlab 可以展示的格式？？？ 
            #-----------------------------------------------------------------------------------
            if 1:

                import matplotlib.pyplot as plt
                #plt.plot(rho_y, rho_x)
                #plt.show()

                # 【1】 一维流形
                #-----------------------------------------------------------------------------
                savename = result_dir+'/test_neural.txt'
                testdata2 = genfromtxt(savename, delimiter=' ')
                coords = testdata2[:, 0:2]

                coords = np.hstack((coords, np.zeros(len(coords)).reshape(-1, 1) ))

                colors = testdata2[:, -3:]

                # plt.cla()
                # for row in range(len(coords)):
                #     plt.plot(coords[row,1], coords[row, 0], color =  tuple(list( colors[row]/255))  )

                # plt.show()


                outdata1 = np.hstack((coords, colors))
                np.savetxt( result_dir + '/mesh_ambient_' + model_name + '.txt', outdata1, delimiter=',')


                # 【2】 一维数据
                #-------------------------------------------------------------------------------
                savename = result_dir+'/latent_neural.txt'
                testdata3 = genfromtxt(savename, delimiter=' ')
                coords = testdata2[:, 0:1]
                coords = np.hstack((coords, np.zeros((len(coords),2)).reshape(-1, 2) ))

                colors = testdata2[:, -3:]
                outdata2 = np.hstack((coords, colors))

                np.savetxt( result_dir + '/mesh_latent_' + model_name + '.txt', outdata2, delimiter=',')




    eee = 999




