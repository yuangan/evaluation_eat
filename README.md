## Efficient Emotional Adaptation for Audio-Driven Talking-Head Generation: Evaluation Instructions

### **Setup**

Before beginning, ensure your environment aligns with the [EAT repository](https://github.com/yuangan/EAT_code) requirements. Install the necessary package:

```bash
pip install face-alignment
```

### Evaluation Instructions

1. **Download Pre-trained Models:** 
   - Access the pre-trained models from [this link](https://drive.google.com/file/d/1qJdAphOQbMTnXTUlv7mMb1kRnCwQ2xDT/view?usp=sharing).
   - After downloading, unzip the files and place them into the `code` folder.

2. **Download Ground Truth Videos:** 
   - Obtain the Ground Truth videos from [this link](https://drive.google.com/file/d/1zMQqb22Lc9ozykcrCjHJ4Hc_Cgom4tHs/view?usp=drive_link).
   - Once downloaded, unzip the files and move them into the root directory (`./`).

3. **Place Your Results:**
   - Position your evaluation results in the `./result` folder.

4. **Execution:**
   - For instance, if your results are located in the folder `./result/deepprompt_eam3d_all_final_313`, execute the following bash command:
    ```
    bash test_psnr_ssim_sync.sh deepprompt_eam3d_all_final_313 0
    ```
The emotion accuracy of EmotionFan will be updated soon.
