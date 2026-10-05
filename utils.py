import numpy as np
import cv2 as cv

def compute_lines(img, vert):
    if vert=='v':
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(1,img.shape[0]//20))
        lines = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel,iterations=1)
        lines = cv.morphologyEx(img,cv.MORPH_OPEN,kernel,iterations=1)
    else:
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(img.shape[1]//30,1))
        lines = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel,iterations=1)
        lines = cv.morphologyEx(img,cv.MORPH_OPEN,kernel,iterations=1)

    contours,_=cv.findContours(lines,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

    lines_list=[]
    for c in contours:
        x,y,w,h = cv.boundingRect(c)
        lines_list.append([x,y,w,h])
    if vert=='v':
        lines_list=sorted(lines_list, key=lambda x: x[0])
    else:
        lines_list=sorted(lines_list, key=lambda x: x[1])

    return lines_list

def compute_lines_pixels(img,vert):
    if vert=='v':
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(1,img.shape[0]//15))
        lines = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel,iterations=1)
        lines_pr = cv.morphologyEx(img,cv.MORPH_OPEN,kernel,iterations=1)
        linues_sum=lines_pr.sum(axis=0)
        ys=np.where(linues_sum>255*img.shape[0]*0.22)[0]
    else:
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(img.shape[1]//15,1))
        lines = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel,iterations=1)
        lines_pr = cv.morphologyEx(img,cv.MORPH_OPEN,kernel,iterations=1)

        linues_sum=lines_pr.sum(axis=1)
        ys=np.where(linues_sum>58_000 )[0]
        
    groups=[]
    for y in ys:
        if not groups or y > groups[-1][-1]+2:
            groups.append([y])
        else:
            groups[-1].append(y)
    
    lines=[]
    for group in groups:
        line=int(np.average(group))
        lines.append(line)
    return lines, lines_pr


def deskew_image(img): 
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    canny=cv.Canny(gray, 100,200)

    lines = cv.HoughLinesP(
        canny,
        rho=1,
        theta=np.pi / 1800,
        threshold = 100,
        minLineLength=300,
        maxLineGap = 30
    )

    angles = []
    # print(lines)
    lines = lines.reshape(-1, 4)
    for x1, y1,x2,y2 in lines:
        angle=np.degrees(np.arctan2(y2-y1,x2-x1))

        if -15 < angle < 15:
            angles.append(angle)

    angle = np.median(angles)
    h,w = img.shape[:2]

    M=cv.getRotationMatrix2D((w/2,h/2), angle, 1.0)

    deskewed = cv.warpAffine(img,M,(w,h),flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return deskewed
def crop_image(img):
    deskewed=deskew_image(img)
    gray=cv.cvtColor(deskewed,cv.COLOR_BGR2GRAY)
    gray=cv.bilateralFilter(gray,25,35,75)
    cv.imwrite('gray_deskewed.png', gray)

    _,tresh=cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)

    h_lines,_=compute_lines_pixels(tresh,'h')
    v_lines,_=compute_lines_pixels(tresh,'v')

    print(len(h_lines))
    print(len(v_lines))

    h_line_top=np.unique(np.array(h_lines))[4]
    h_line_bottom=np.unique(np.array(h_lines))[4+12]
    v_line_left=np.unique(np.array(v_lines))[1]
    v_line_right=np.unique(np.array(v_lines))[1+20]

    cropped_img = tresh[h_line_top:h_line_bottom,v_line_left:v_line_right]
    cropped_img_org = deskewed[h_line_top:h_line_bottom,v_line_left:v_line_right]

    return cropped_img, cropped_img_org

def find_cells(h_lines,v_lines):
    cells=[]
    for h in range(len(h_lines)-1):
        row=[]
        for v in range(len(v_lines)-1):
            x1,y1=v_lines[v],h_lines[h]
            x2,y2=v_lines[v+1],h_lines[h+1]
            row.append([x1,y1,x2,y2])
        cells.append(row)

    return cells
        