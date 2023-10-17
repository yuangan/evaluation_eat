## Benchmark and Evaluation of "Efficient Emotional Adaptation for Audio-Driven Talking-Head Generation"

### **Setup**

Before beginning, ensure your environment aligns with the [EAT repository](https://github.com/yuangan/EAT_code) requirements. Install the necessary package:

```bash
pip install face-alignment
```

### Evaluation Instructions

1. **Download Pre-trained Models:** 
   - Access the pre-trained models from [this link](https://drive.google.com/file/d/1qJdAphOQbMTnXTUlv7mMb1kRnCwQ2xDT/view?usp=sharing).
   - After downloading, unzip the files and place them into the `code` folder.
  
     ```unzip code.zip -d code```

2. **Download Ground Truth Videos:** 
   - Obtain the cropped Ground Truth videos from [this link](https://drive.google.com/file/d/1zMQqb22Lc9ozykcrCjHJ4Hc_Cgom4tHs/view?usp=drive_link).
   - Once downloaded, unzip the files and move them into the root directory with the command:
     
     ```unzip talking_head_testing.zip -d talking_head_testing```

3. **Place Your Results:**
   - Position your evaluation results in the `./result` folder.

4. **Execution:**
   - For instance, if your sampled (100) test results are located in the folder `./result/deepprompt_eam3d_all_final_313`, execute the following bash command:
     
    ```
    bash test_psnr_ssim_sync_emoacc.sh deepprompt_eam3d_all_final_313 0
    ```
   - If you want to test the whole 985 results in MEAD test set, execute the following bash command:
    
    ```
    bash test_psnr_ssim_sync_emoacc_985.sh deepprompt_eam3d_all_final_313_985 0
    ```
