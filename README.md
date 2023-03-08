# AIFFEL Thon 

## Project
- 이미지와 텍스트를 모두 처리할 수 있는 생성형 멀티모달 챗봇

### 내가 수행한 Task
- 차량파손 종류와 위치 탐지 모델 

### 쏘카 차량 파손 데이터
- Dent(찌그러짐)
- Scratch(긁힘)
- Spacing(이격)


### 사용 모델 
- U-Net++
- U-Net++ 사용 이유 : 적은 데이터, 속도도 빠르다.  
- unet++는 skip pathways에서 dense convolution block 들을 통해 추출된 feature map들이 unet에 비해서 fine-grained detail을 더 잘 포착할 수 있도록 합니다.
- 파손 종류별로 하나씩 모델을 만듬


### Image Crop
- 파손 영역만 crop하여 학습
- IoU score 3% 개선


### Dual Channel training
- 일반적인 semantic segmentation 방식으로는 binary classification 문제에서 "파손인가 아닌가?" 만을 탐지
- 이 방식을 개선시켜 back-ground와 fore-ground를 분리하여 학습 
- 이렇게 할 경우 손실되는 정보량이 줄어들어 성능이 개선될 가능성이 있음
- 실제로 iou score가 30% 가량 개선





