# Introduction
本次作業練習安裝Detectron2 v0.5，透過安裝過程實踐解決環境依賴產生問題，並改寫config檔案進行foggy_cityscape資料集的訓練，最後撰寫一個用於inference檔案
# 一、環境安裝步驟說明
本次作業使用Nvidia RTX 4090進行，搭載Ubuntu 22.04作業系統，CUDA Version為12.4。
## 1.檢查cuda
[cuda12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)
```
nvcc -v
```

## 2.torch安裝
根據官方的文件說明，原本選擇使用cuda11.1.0+torch1.8.0進行環境部署；嘗試後發現1.8.0會導致錯誤，改用cuda12.4、torch2.4.1。
```
conda create -n detectron python=3.11
conda activate detectron

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

```
驗證環境
```
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```
2.4.1+cu124
## 3.確認gcc環境
```
gcc --version
```
gcc (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0
## 4.Detectron 2安裝
```
pip install git+https://github.com/facebookresearch/detectron2.git@v0.5
```
## 5.下載資料集
(1) NAS掛載到本地
(2) scp -r `本地nas要上傳資料集的路徑` `username@remote_server:遠端datasets路徑`
(3) `tar -xvf tar_path` ：解壓縮資料集
## 6.問題解決
(1) RuntimeError: radix_sort: failed on 1st step: cudaErrorInvalidDevice: invalid device ordinal
嘗試python train_net.py --config configs/city_resnet50.yaml --num-gpus 1仍繼續報錯，後經過查詢發現是1.8.x導致的bug，同時發現環境沒有安裝CUDA。
[參考來源](https://blog.csdn.net/qq_55068938/article/details/121270986)

(2) 執行安裝Detectron2時遇到CUDA version mismatches
    [解決方法](https://blog.csdn.net/m0_51516317/article/details/139423784)將報錯的code改成pass
    
(3) AttributeError: module 'PIL.Image' has no attribute 'LINEAR'. Did you mean: 'BILINEAR'
    降低pillow版本
    ```pillow==9.5.0```
# 二、模型訓練與推論結果
模型使用Coco format evaluator去評估模型效能，AP數值越高代表偵測結果越精準，AP50則是預測框和gt有一半重疊即算正確，而APs、APm、APl則代表模型對於小、中、大物體的預測平均精確度。
## 1.Cityscapes dataset
使用原始給定的參數，iteration為24000，並在每訓練8000 iteration後進行驗證集測試。
![image](https://hackmd.io/_uploads/S1-BnS3Yxe.png)
訓練時的total_loss:
![image](https://hackmd.io/_uploads/SJcZO8hYel.png)
![image](https://hackmd.io/_uploads/H1Pa7HnKxe.png)


## 2.foggy_cityscape dataset
Config檔中的參數，考量到iter:24000的訓練時長過長(大約5小時)，嘗試縮減至5000，訓練時長確實減少為1小時；同時將STEPS同步減小，讓模型在訓練至iter=1800時能夠同步減小學習率，並將EVAL_PERIOD設為1000，讓他在訓練1000 iteration後使用驗證集驗證一次。
![image](https://hackmd.io/_uploads/rJ5p_HhFex.png)
訓練時的total_loss:
![image](https://hackmd.io/_uploads/H18muL3tee.png)

![image](https://hackmd.io/_uploads/ByLiHShtxl.png)

![image](https://hackmd.io/_uploads/HyTUsUhKxl.png)

修改config檔參數後再次進行訓練，這次僅將MAX_ITER更改為3000後再次進行訓練。可以透過AP發現模型並沒有出現很大的退化，再加上iter5000的AP折線圖能觀察到AP在iter 3000後就維持在35，若要有更好的表現需要透過調整其他參數，或是增大MAX_ITER。

![image](https://hackmd.io/_uploads/HJSHyUhKeg.png)
![image](https://hackmd.io/_uploads/SJp4xUhKlg.png)

![image](https://hackmd.io/_uploads/ryYV_UnFel.png)

# 三、inference
此部分參考Detectron2官方提供的demo.py並進行簡化，執行時透過`--config`、`--weights`、`--image_path`，可以進行單張圖片的推理。
1.Cityscapes dataset_24000
![image](https://hackmd.io/_uploads/rygqxUnFgl.png)
2.foggy_cityscapes_5000
![image](https://hackmd.io/_uploads/rJ2VYL3Fgl.png)

3.foggy_cityscapes_3000
![image](https://hackmd.io/_uploads/rJBqFInFxg.png)

# 四、Discussion
兩次foggy資料集的訓練結果在AP上沒有明顯差異，都差不多35。觀察各類別的AP能看到，iter 5000只有train略高於3000，反而3000在bus、motocycle的表現略好一點，尤其是motocycle從23.9轉變為26.1。反觀使用iter 24000進行訓練的Cityscapes 資料集，AP有提升至40.3，loss降到0.5以下，感覺這類型資料需要透過較多的iteration和更小的batch_size才能學習得更好。

這次的環境部屬反而更依賴CSDN上的討論與經驗帖，使用AI進行輔助反而一直受限於官方文件推薦的版本搭配，導致遇到了AI推薦將Nvidia驅動刪除並重裝，最後出現驅動和底層驅動版本衝突的荒謬故事。並意外發現較高版本的cuda、pytorch對於安裝上的影響並沒想像中的高，最多會遇到相關套件版本太高導致某些方法被官方棄用而出現的報錯。
# 其他參考資料
https://blog.csdn.net/stevenZXZ/article/details/133864392
github.com/facebookresearch/detectron2