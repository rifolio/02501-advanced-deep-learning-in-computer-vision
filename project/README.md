# Project 2 - Video classification

**Requirements**
    - Add the data and extract it `unzip ucf101.zip`. It will be saved in the directory ufc10 which is gitignored. DO NOT PUSH THE DATA!

**Data**
There are 10 classes of actions:
 0. BodyWeightSquats
 1. HandstandPushups
 2. HandstandWalking
 3. JumpingJack
 4. JumpRope
 5. Lunges
 6. PullUps
 7. PushUps
 8. TrampolineJumping
 9. WallPushups

**Running batch job**

```
bsub < jobscript.sh
```

## 4.1 Explore models
- Aggregate single frames
- Early fusion
- Late fusion
- 3D CNN

## 4.2 Leakage between train/test/validation


## 4.3 Dual-stream networks



# Misc
- Taking inspiration from structure: https://www.geeksforgeeks.org/deep-learning/how-to-structure-a-pytorch-project/
- Rich logging (handler): https://github.com/Textualize/rich/discussions/1173