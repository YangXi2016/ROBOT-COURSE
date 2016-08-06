# -*- encoding: UTF-8 -*-
''' 文件原名：Versionpy.py'''
''' 
函数使用格式：
找到乒乓球的X，Y坐标：                
    [x,y] = FindBall(IP)
找到箱子的X，Y坐标和面与乒乓球的夹角
    [x,y,theta] = FindBox(IP)
单位：cm, rad
'''

from naoqi import ALProxy
from PIL import Image      
import vision_definitions
import numpy as np
import sympy
import cv2
import math

# 供外部调用的函数，返回当前机器人看到的乒乓球的位置(x,y)
def FindBall(IP):
 [cameraX,cameraY,cameraType] = get_coordinates(IP);
 print "Find ball in camera complete!"
 if cameraType!=-1:
   [realX,realY] = Camera2Real(cameraX,cameraY,cameraType);
   return [round(realX/10),round(realY/10)]
 else:
   return [-1,-1] 

# 供外部调用的函数，返回当前机器人看到的箱子的位置(x,y)和法向量角度theta
def FindBox(IP):
 [x,y]=FindBall(IP)
 print "Find ball in camera complete!"
 [nx,ny,cameraType] = get_coordinates_box(IP); #现在的nx其实是tan theta
 print "Find box in camera complete!"
 
 # print "Ball x y"
 # print [x,y]
 if (x!=-1) and (nx!=0):
   # [realX,realY] = Camera2Real(x,y,cameraType);
   realX = x
   realY = y
   print "---------"
   res_nx = nx+3.14/2
   if(res_nx>3.14/2):
			res_nx=res_nx-3.14
   return [round(realX),round(realY),round((res_nx+0.1)*100)/100]
 else:
   return [-1,-1,-1] 

# 子函数，最小二乘法
def least_square(x,y):
    n=len(x)
    integer_x=0
    integer_x2=0
    integer_xy=0
    integer_y=0
    for i in x:
        integer_x+=i
        integer_x2+=i*i
    for k in y:
        integer_y+=k
    for i,m in enumerate(x):
        integer_xy+=m*y[i]
    a=(integer_xy-(integer_x*integer_y)/n)/(integer_x2-(integer_x*integer_x/n))
    b=integer_y/n-a*integer_x/n
    return a,b

# 从一张图片中准确筛选出箱子上的“标定线”，检测到则返回[realX,realY,nx,ny,nz]方向，没有则返回-1	
def get_box(openname,cameraID):       
    '''检测直线'''
    # openname='temp1.png'
    standard_line=[]
    realX_list=[]
    realY_list=[]
    '''
    #白色的hsv范围
    low_h=0
    high_h=180
    low_s=0
    high_s=100
    low_v=150
    high_v=255
    '''
    #青色的hsv范围
    # zheliyaogai
    hh = 55
    low_h=hh-20
    high_h=hh+20
    low_s=25
    high_s=225
    low_v=25
    high_v=225
    kernel=np.ones((2,7),np.uint8)
	
    Origin_img=cv2.imread(openname)                             #Origin_img:原始图像
    Origin_img=cv2.blur(Origin_img,(5,5))
    '''	
    cv2.imshow('mask', Origin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    Origin_hsv=cv2.cvtColor(Origin_img,cv2.COLOR_BGR2HSV)
    lower_yellow=np.array([low_h,low_s,low_v])
    upper_yellow=np.array([high_h,high_s,high_v])
    mask=cv2.inRange(Origin_hsv,lower_yellow,upper_yellow)      #mask:颜色滤波器
    '''
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)            #开运算后
    '''
    zcv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    canny=mask
    #canny = cv2.Canny(mask, 32, 100)
    '''
    cv2.imshow('canny', canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    result=Origin_img.copy()
    #标准霍夫直线变化
    lines = cv2.HoughLines(canny,1,np.pi/180,120)
    if lines ==None:
        print 'no line 1'
        return [-1,-1,-1]
    
    for line in lines[:,0,:]: 
        rho = line[0] #第一个元素是距离rho  
        theta= line[1] #第二个元素是角度theta  
        #print rho  
        #print theta
        
        if  (theta < (3*np.pi/8. )) or (theta > (5.*np.pi/8.0)): #垂直直线  图片坐标系右正x下负y
            continue
            '''
            #该直线与第一行的交点  
            pt1 = (int(rho/np.cos(theta)),0)  
            #该直线与最后一行的焦点  
            pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])  
            #绘制一条白线  
            cv2.line( result, pt1, pt2, (255))
            '''
            
        else: #水平直线
            pt1 = (0,int(rho/np.sin(theta)))  
            #该直线与最后一列的交点  
            pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
            flag=True			
            if cameraID==0:
                flag=(((pt1[1]+pt2[1])/2)>280)

            
            if (flag):
                #绘制一条直线  
                #cv2.line(result, pt1, pt2, (255), 1)
                standard_line.append(line)

    if len(standard_line)==0:
        print 'no line 2'
        return [-1,-1,-1]
    rho_sum=0
    theta_sum=0
    for line in standard_line:
        rho_sum+=line[0]
        theta_sum+=line[1]
        
    rho=rho_sum/len(standard_line)
    theta=theta_sum/len(standard_line)
    # 该直线与第一列的交点  
    pt1 = (0,int(rho/np.sin(theta)))  
    #该直线与最后一列的交点  
    pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))  
    #绘制一条直线  
    cv2.line(result, pt1, pt2, (255,0,0), 5)

    '''cv2.imshow('houghlines', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    cv2.imwrite('houghlines5.jpg',result)
    
    print '?'
    dot_num=50
    if pt1[1]<0:
        y_list=np.linspace(pt2[1]/3,2*pt2[1]/3,dot_num)
    else:
        y_list=np.linspace((pt2[1]-pt1[1])/3,2*(pt2[1]-pt1[1])/3,dot_num)

    x_list=(rho-y_list*math.sin(theta))/math.cos(theta)

    for i in range(dot_num):
        [realX,realY]=Camera2Real_box(x_list[i],y_list[i],cameraID)
        realX_list.append(realX)
        realY_list.append(realY)
    #y=np.linespace
    #x=(rho-ysin(theta))/cos(theta)

    (a,b)=least_square(realX_list,realY_list)
	
    realTheta=math.atan(a)
    n=[realTheta,0]
    '''检测直线'''

    # 通过在箱子下面粘贴乒乓球，使用FindBall来避免误差
    '''利用圆检测中心位置（圆心并不在直线上，这里只确定x坐标，y坐标根据直线来。）'''
    '''
    minDist=50
    MINRADIUS=101
    MAXRADIUS=70

    peaks=[]  #用于存放四个蓝色的圆信息
    #蓝色的hsv范围
    low_h=90
    high_h=150
    low_s=80
    high_s=255
    low_v=80
    high_v=255

    Origin_img=cv2.imread(openname)                             #Origin_img:原始图像
    cv2.imshow('Origin_image',Origin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    Origin_hsv=cv2.cvtColor(Origin_img,cv2.COLOR_BGR2HSV)
    lower_yellow=np.array([low_h,low_s,low_v])
    upper_yellow=np.array([high_h,high_s,high_v])
    mask=cv2.inRange(Origin_hsv,lower_yellow,upper_yellow)      #mask:颜色滤波器

    cv2.imshow('mask',mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


        # 对原图像和掩模进行位运算
    res=cv2.bitwise_and(Origin_img,Origin_img,mask=mask)        #res:经颜色滤波后的图像

    #cv2.imwrite("temp2.png",res)

    Gray_img=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)               #Gray_img:res转换为Gray格式的图像，作为霍夫变化的参数
    Gray_img = cv2.medianBlur(Gray_img,5)
    cimg = cv2.cvtColor(Gray_img,cv2.COLOR_GRAY2BGR)
    circels=None
    circles = cv2.HoughCircles(Gray_img,cv2.cv.CV_HOUGH_GRADIENT,1,minDist,param1=100,param2=10,minRadius=MINRADIUS,maxRadius=MAXRADIUS)        #该组参数比较适合检测较小被遮挡的乒乓
    print 'search circles'

    if circles==None:
        print 'no circles 1'
        return [-1,-1,0,0,0]
    position_x=0
    for i in circles[0,:]:
        if(mask[i[1],i[0]]!=0):
            #draw the outer circle
            cv2.circle(result,(i[0],i[1]),i[2],(0,255,0),2)
            #draw the center of the circle
            cv2.circle(result,(i[0],i[1]),2,(0,0,255),3)
            position_x+=i[0]
            peaks.append(i)            

    if position_x==0:
        print 'no circles 2'
        return [-1,-1,0,0,0]
    
    position_x=position_x/len(peaks)
    position_y=(rho-position_x*math.cos(theta))/math.sin(theta)
	'''
    '''利用圆检测中心位置'''
    '''cv2.imshow('houghlines', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    cv2.imwrite('houghlines5.jpg',result)
    #print [position_x,position_y,n[1],n[2]]
    return [n[0],n[1],cameraID]

# 判断变量V是被被定义    
def isset(v):
    try:
        type(eval(v))
    except:
        return 0
    else:
        return 1

# 从一张图片中准确筛选出在地面上的最近的乒乓球
def get_pingpong(openname,cameraID):
    circle_x=-1
    circle_y=-1
    
    low_h=3
    high_h=30
    low_s=80
    high_s=235
    low_v=80
    high_v=235

    if(cameraID==0):        #up_camera
        MINRADIUS=2
        MAXRADIUS=35
    elif(cameraID==1):      #down_camera
        MINRADIUS=10
        MAXRADIUS=50
    else:
        print "get_pingpong函数缺少cameraID参数";
        exit;

    
    Origin_img=cv2.imread(openname)                             #Origin_img:原始图像

    Origin_hsv=cv2.cvtColor(Origin_img,cv2.COLOR_BGR2HSV)
    lower_yellow=np.array([low_h,low_s,low_v])
    upper_yellow=np.array([high_h,high_s,high_v])
    mask=cv2.inRange(Origin_hsv,lower_yellow,upper_yellow)      #mask:颜色滤波器


    # 对原图像和掩模进行位运算
    res=cv2.bitwise_and(Origin_img,Origin_img,mask=mask)

    '''
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    Gray_img=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)               #Gray_img:res转换为Gray格式的图像，作为霍夫变化的参数
    Gray_img = cv2.medianBlur(Gray_img,5)
    cimg = cv2.cvtColor(Gray_img,cv2.COLOR_GRAY2BGR)
    circels=None
    circles = cv2.HoughCircles(Gray_img,cv2.cv.CV_HOUGH_GRADIENT,1,3,param1=100,param2=10,minRadius=MINRADIUS,maxRadius=MAXRADIUS)        
    #该组参数比较适合检测较小被遮挡的乒乓
    print "search pingpong in res_image";
    if circles==None:
        Gray_img=cv2.cvtColor(Origin_img,cv2.COLOR_BGR2GRAY)               
        cimg = cv2.cvtColor(Gray_img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(Gray_img,cv2.cv.CV_HOUGH_GRADIENT,1,10,param1=100,param2=10,minRadius=MINRADIUS,maxRadius=MAXRADIUS)
        print "search pingpong in origin_image"
    if circles==None:
        print "no pingpong1"
        return [-1,-1]  
        
    circles = np.uint16(np.around(circles))
    # print circles
    for i in circles[0,:]:
        if(mask[i[1],i[0]]!=0):
            if(cameraID==0):
                standard_r=6.965e-06*pow(i[1],2.245)*1#ratio1
                if(standard_r<8):
                    flag=i[2]<standard_r*1.2
                else:
                    flag=(i[2]>standard_r*0.8)and(i[2]<standard_r*1.2)
            elif(cameraID==1):
                standard_r=0.038*i[1]+7#(21+i[1]/32)*1#ratio2
                '''print "========"
                print standard_r
                print i[2]
                print "========"'''
                flag=(i[2]>standard_r*0.5)and(i[2]<standard_r*1.5)
            if(flag):
                #draw the outer circle
                cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                #draw the center of the circle
                cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
                if circle_y<i[1]:
                    circle_x=i[0]  #x
                    circle_y=i[1]  #y
                    circle_r=i[2]  #r
    '''       
    cv2.imshow('detected circles',cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print circle_x,circle_y
    '''
    cv2.imwrite("temp_pingpong.png",cimg)
    return circle_x,circle_y

# 从Nao机器人中读取并储存照片
def get_coordinates(IP):
    temp0name="temp0.png"
    temp1name="temp1.png"

    #IP = "192.168.1.1"  # Replace here with your NAOqi's IP address.
    PORT = 9559

    ####
    # Create proxy on ALVideoDevice

    #print "Creating ALVideoDevice proxy to ", IP

    camProxy = ALProxy("ALVideoDevice", IP, PORT)

    ####
    # Register a Generic Video Module

    #resolution = vision_definitions.kQVGA
    #colorSpace = vision_definitions.kYUVColorSpace
    resolution =3 #vision_definitions.kQQVGA
    colorSpace =11
    fps = 30

    cameraID = 1        #cameraButton
    camProxy.setParam(vision_definitions.kCameraSelectID,cameraID)
    nameId = camProxy.subscribe("python_GVM", resolution, colorSpace, fps)
    #print nameId
    '''
    print 'getting images in local'
    for i in range(0, 20):
      camProxy.getImageLocal(nameId)
      camProxy.releaseImage(nameId)
    '''

    camProxy.setResolution(nameId, resolution)


    print 'getting images in remote'
    #for i in range(0, 20):
    img1=camProxy.getImageRemote(nameId)
    camProxy.unsubscribe(nameId)

     # Create a PIL Image from our pixel array.
    im1 = Image.fromstring("RGB", (img1[0], img1[1]), img1[6])

      # Save the image.
    im1.save(temp0name, "PNG")
    
    print "search pingpong CameraBottom"
    [circle_x,circle_y]=get_pingpong(temp0name,cameraID)
    
    #print circle_x
    
    '''if circle_x==-1:
        print "search pingpong CameraTop"
        cameraID=0                      #CameraTop
        camProxy.setParam(vision_definitions.kCameraSelectID,cameraID)
        nameId = camProxy.subscribe("python_GVM", resolution, colorSpace, fps)
        #print nameId

        img2=camProxy.getImageRemote(nameId)

        camProxy.unsubscribe(nameId)

         # Create a PIL Image from our pixel array.
        im2 = Image.fromstring("RGB", (img2[0], img2[1]), img2[6])

          # Save the image.
        im2.save(temp1name, "PNG")        
        [circle_x,circle_y]=get_pingpong(temp1name,cameraID)    '''

    if circle_x==-1:
        print "No pingpong"
        return [-1,-1,-1]
    else:
        print [circle_x,circle_y,cameraID]
        return [circle_x,circle_y,cameraID]

# 从Nao机器人中读取并储存照片，并找到箱子
def get_coordinates_box(IP):
    temp0name="temp0_box.png"
    temp1name="temp1_box.png"

    #IP = "192.168.1.1"  # Replace here with your NAOqi's IP address.
    PORT = 9559

    ####
    # Create proxy on ALVideoDevice

    #print "Creating ALVideoDevice proxy to ", IP

    camProxy = ALProxy("ALVideoDevice", IP, PORT)

    ####
    # Register a Generic Video Module

    #resolution = vision_definitions.kQVGA
    #colorSpace = vision_definitions.kYUVColorSpace
    resolution =3 #vision_definitions.kQQVGA
    colorSpace =11
    fps = 30

    cameraID = 0        #cameraButton
    camProxy.setParam(vision_definitions.kCameraSelectID,cameraID)
    nameId = camProxy.subscribe("python_GVM", resolution, colorSpace, fps)
    #print nameId
    '''
    print 'getting images in local'
    for i in range(0, 20):
      camProxy.getImageLocal(nameId)
      camProxy.releaseImage(nameId)
    '''

    camProxy.setResolution(nameId, resolution)


    print 'getting images in remote'
    #for i in range(0, 20):
    img1=camProxy.getImageRemote(nameId)
    camProxy.unsubscribe(nameId)

     # Create a PIL Image from our pixel array.
    im1 = Image.fromstring("RGB", (img1[0], img1[1]), img1[6])

      # Save the image.
    im1.save(temp0name, "PNG")
    
    print "search box CameraBottom"
    [nx,ny,cameraID]=get_box(temp0name,cameraID)
    
    #print circle_x
    if cameraID==-1:
        print "search box CameraTop"
        cameraID=1                      #CameraTop
        camProxy.setParam(vision_definitions.kCameraSelectID,cameraID)
        nameId = camProxy.subscribe("python_GVM", resolution, colorSpace, fps)
        #print nameId

        img2=camProxy.getImageRemote(nameId)

        camProxy.unsubscribe(nameId)

         # Create a PIL Image from our pixel array.
        im2 = Image.fromstring("RGB", (img2[0], img2[1]), img2[6])

          # Save the image.
        im2.save(temp1name, "PNG")
        
        [nx,ny,cameraID]=get_box(temp1name,cameraID)

    if cameraID==-1:
        print "No box"
        return [0,0,0]
    else:
        #print [x,y,nx,ny]
        return [nx,ny,cameraID]

# 子函数，主要被函数FindBall调用，从乒乓球像素点的位置(x,y)返回真实世界乒乓球的坐标(X,Y)
def Camera2Real(x,y,camera):
# up camera, camera == 0
# buttom camera, camera == 1
    if camera==0:
        # [R;t]
        Rt = np.array([[0.0015,0.9878,-0.1556],[-0.9999,0.0030,0.0096],[0.01,0.1556,0.9878],[-42.8895,38.8069,2119.2]])
        K = np.array([[1101.1,0,0],[0,1094.2,0],[651.3354,478.8062,1]])
        s = 30
        M = np.dot(Rt,K)/s
        N = np.array( [row[i] for i in range(0, 1) for row in M] )
        # 这个数找weijie
        X = 795
        '''
        # 这两个数找xishen
        x = 659
        y = 931
        '''
        # 解线性方程组
        solve_a = np.array([[M[0,0]+M[3,0]/X,M[1,0],M[2,0]],[M[0,1]+M[3,1]/X,M[1,1],M[2,1]],[M[0,2]+M[3,2]/X,M[1,2],M[2,2]]])
        solve_b = np.array([[x],[y],[1]])
        output = np.linalg.solve(solve_a,solve_b)
        '''
        print 'X='+'%f' %X
        print 'Y='+'%f' %(X/output[0]*output[1])
        print 'Z='+'%f' %(X/output[0]*output[2])
        '''
        L = 2100              
        returnX = (X/output[0]*output[2]) +  L + 2
        returnY = (X/output[0]*output[1]) + 55
        '''
        print '---RESULT---'
        print returnX
        print returnY
        '''
        return np.array([returnX,returnY])

    if camera==1:
        # [R;t]
        Rt = np.array([[0.9985,-0.0004,0.0545],[0.0492,0.4368,-0.8982],[-0.0235,0.8996,0.4361],[-51.4210657045367,149.135309293409,859.972441430778]])
        K = np.array([[1183.82122082327,0,0],[0,1179.97372207552,0],[623.778434097122,349.785531449855,1]])
        '''s = 30'''
        M = np.dot(Rt,K)
        N = np.array( [row[i] for i in range(0, 1) for row in M] )
        # 这个数找weijie
        Z = -20
        # 这两个数找xishen
        '''
        x = 550
        y = 532
        '''
        # 解线性方程组
        solve_a = np.array([[M[0,0],M[1,0],M[2,0]+M[3,0]/Z],[M[0,1],M[1,1],M[2,1]+M[3,1]/Z],[M[0,2],M[1,2],M[2,2]+M[3,2]/Z]])
        solve_b = np.array([[x],[y],[1]])
        output = np.linalg.solve(solve_a,solve_b)
        '''
        print 'X='+'%f' %(Z/output[2]*output[0])
        print 'Y='+'%f' %(Z/output[2]*output[1])
        print 'Z='+'%f' %Z
        '''
        L = 600
        returnX = L-(Z/output[2]*output[1]) + 0.5
        returnY = -(Z/output[2]*output[0]) - 2.1
        '''
        print '---RESULT---'
        print returnX
        print returnY
        '''
        return np.array([returnX,returnY])

# 子函数，主要被函数FindBox调用，从箱子的定位直线上的某个像素点的位置(x,y)返回真实世界该点坐标(X,Y)
def Camera2Real_box(x,y,camera):
# up camera, camera == 0
# buttom camera, camera == 1
    if camera==0:
        # [R;t]
        Rt = np.array([[0.0015,0.9878,-0.1556],[-0.9999,0.0030,0.0096],[0.01,0.1556,0.9878],[-42.8895,38.8069,2119.2]])
        K = np.array([[1101.1,0,0],[0,1094.2,0],[651.3354,478.8062,1]])
        s = 30
        M = np.dot(Rt,K)/s
        N = np.array( [row[i] for i in range(0, 1) for row in M] )
        # 这个数找weijie
        X = 795-(337-18)
        '''
        # 这两个数找xishen
        x = 659
        y = 931
        '''
        # 解线性方程组
        solve_a = np.array([[M[0,0]+M[3,0]/X,M[1,0],M[2,0]],[M[0,1]+M[3,1]/X,M[1,1],M[2,1]],[M[0,2]+M[3,2]/X,M[1,2],M[2,2]]])
        solve_b = np.array([[x],[y],[1]])
        output = np.linalg.solve(solve_a,solve_b)
        '''
        print 'X='+'%f' %X
        print 'Y='+'%f' %(X/output[0]*output[1])
        print 'Z='+'%f' %(X/output[0]*output[2])
        '''
        L = 2100              
        returnX = (X/output[0]*output[2]) +  L + 2
        returnY = (X/output[0]*output[1]) + 55
        '''
        print '---RESULT---'
        print returnX
        print returnY
        '''
        return np.array([returnX,returnY])

    if camera==1:
        # [R;t]
        Rt = np.array([[0.9985,-0.0004,0.0545],[0.0492,0.4368,-0.8982],[-0.0235,0.8996,0.4361],[-51.4210657045367,149.135309293409,859.972441430778]])
        K = np.array([[1183.82122082327,0,0],[0,1179.97372207552,0],[623.778434097122,349.785531449855,1]])
        '''s = 30'''
        M = np.dot(Rt,K)
        N = np.array( [row[i] for i in range(0, 1) for row in M] )
        # 这个数找weijie
        Z = -20-(337-18)
        # 这两个数找xishen
        '''
        x = 550
        y = 532
        '''
        # 解线性方程组
        solve_a = np.array([[M[0,0],M[1,0],M[2,0]+M[3,0]/Z],[M[0,1],M[1,1],M[2,1]+M[3,1]/Z],[M[0,2],M[1,2],M[2,2]+M[3,2]/Z]])
        solve_b = np.array([[x],[y],[1]])
        output = np.linalg.solve(solve_a,solve_b)
        '''
        print 'X='+'%f' %(Z/output[2]*output[0])
        print 'Y='+'%f' %(Z/output[2]*output[1])
        print 'Z='+'%f' %Z
        '''
        L = 600
        returnX = L-(Z/output[2]*output[1]) + 0.5
        returnY = -(Z/output[2]*output[0]) - 2.1
        '''
        print '---RESULT---'
        print returnX
        print returnY
        '''
        return np.array([returnX,returnY])

# 测试用的代码
if __name__=='__main__':
    #[x,y,xx,yy] = FindBox('169.254.184.149')
    [x,y,t] = FindBox("169.254.194.202")
    print '================'
    print [x,y,t]
    #print [theta]
