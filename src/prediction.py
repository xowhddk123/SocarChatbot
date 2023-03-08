import io
import base64
from cv_lib import cv2, nn, plt, torch, os, cm, np, Image
from Utils import main_path, model_path, device
from Models import Unet

if __name__ =="__main__":
    num = "00"
    def detection(input_path, dent, scratch, spacing):
        img = cv2.imread(input_path)   
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512))
        
        img_input = img / 255.
        img_input = img_input.transpose([2, 0, 1])
        img_input = torch.tensor(img_input).float().to(device)
        img_input = img_input.unsqueeze(0)
        
        m = nn.Threshold(0.3, 0)
        
        fig, ax = plt.subplots(1,4, figsize = (10,3))
        
        ax[0].imshow(img)
        ax[0].axis('off')
        
        models = [dent, scratch, spacing]    
        labels = ['dent', 'scratch', 'spacing']
        img_outputs = []
        for i, (label, model) in enumerate(zip(labels, models), 1):
            output = model(img_input)
            output = torch.sigmoid(output)
            output = m(output)
            
            
            img_output = torch.argmax(output, dim = 1).detach().cpu().numpy()
            img_output = img_output.transpose([1,2,0])
            img_outputs +=[img_output]
            
            ax[i].set_title(label)
            ax[i].imshow(img, alpha = 0.5)
            ax[i].imshow(img_output, cmap = cm.gray, alpha = 0.5)
            ax[i].axis('off')
        fig.set_tight_layout(True)

        with io.BytesIO() as rawBytes:
            fig.savefig(rawBytes, format="png", dpi=128)
            np.frombuffer(rawBytes.getvalue(), dtype=np.uint8)
            rawBytes.seek(0)
            base64_img = base64.b64encode(rawBytes.read()).decode("utf-8")
            "data:image/png;base64," + base64_img
            rawBytes.close()
            Image.open(io.BytesIO(base64.b64decode(base64_img))).save("aaa.png")
        
        for label,output in zip(labels,img_outputs):
            if output.sum() > 0:
                output = np.concatenate([output]*3,axis=2)
                array = np.reshape(output, (512, 512, 3))
                data = Image.fromarray(np.uint8(array * 255))
                data.save('gfg_dummy_pic.png')
                print(f'해당 차량에 {label} 파손이 {output.sum()}만큼 감지 됩니다.')
        # return outputs
        
    # input path는 임의로 받아서 넣어야함    
    input_path = os.path.join(main_path, 'dent/test/images/20190226_13402_20349388_89d790ad7c7813747f0be995df66d0de.jpg')

    dent = torch.load(os.path.join(model_path, f'dent{num}.pt'), map_location=device)   # 저장되어있는 모델 load
    scratch = torch.load(os.path.join(model_path, f'scratch{num}.pt'), map_location=device) 
    spacing = torch.load(os.path.join(model_path, f'spacing{num}.pt'), map_location=device) 
    
    
    # 파손 탐지 함수 실행
    detection(input_path, dent, scratch, spacing)