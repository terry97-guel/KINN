
# Kinematics-Informed Neural Networks: A Unified Model for Soft and Rigid Robots

## DEMO - Painting Letters with Soft Finger and Rigid Arm
https://github.com/terry97-guel/KINN/assets/80967532/f968d812-89c2-4f8d-a142-0ebb6afa06d2



## DEMO - Opening Bottle with Soft Finger and Rigid Arm
https://github.com/terry97-guel/KINN/assets/80967532/87474e84-ab25-408e-867b-7042b3d5fb0e

## Configuration of the soft robot
https://github.com/terry97-guel/KINN/assets/80967532/7cb24b0f-27bc-4bd9-a8c3-783100cbc5de

## Requirements
Python packages
```
pip3 install -r requirements.txt
```

[Optional-1]
To control robots, install [ROS-Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
Codes to control the soft fingers and UR-5 can be found [HERE](https://github.com/terry97-guel/SORO-Dynamixel-python)


[Optional-2]
To visulize the motion planning, install [RVIZ](http://wiki.ros.org/rviz/UserGuide)

## Train (with out Discrete Optimization)
Training the model for given discrete parameters.
Training configs can be found in `code/configs` and, use arguments to choose model and dataset.


```
python3 code/main.py --configs "${MODEL}/${DATASET}.py"
```
Choose, 
`${MODEL}` from `[FC_PRIMNET, PRIMNET, PCC_PRIMNET, PRIMNET_FULL]`
`${DATASET}` from `[FINGER, ABAQUS_32, ELASTICA]`
(e.g. To train `PureNN` on `FINGER` dataset,  `python3 code/main.py --configs "PRIMNET/FINGER.py"`)


## Train (with Discrete Optimization)
```
python3 code/main_kas.py --configs "${MODEL}/${DATASET}.py" --generation_num 100 --agent_num 50 --epochs 1000
```

## Control - Following Square Trajectories
```
cd code/control
python3 Experiment_1.py
```

## Control - Painting Letters
```
cd code/control
python3 Experiment_2.py
```

## Control - Opening Bottles
Experiment:Opening Bottles uses Pre-trained [Segmentation module](https://github.com/terry97-guel/UnseenObjectClustering) to locate the bottles to open.

```
cd code/control
python3 Experiment_3.py
```

## Dataset
We test out method on three dataset,

`ABAQUS`: Pneumatic Robot (Simuation, [ABAQUS](https://www.3ds.com/products-services/simulia/products/abaqus/))
`ELASTICA`: Tendon-Driven Robot (Simuation, [Elastica](https://github.com/GazzolaLab/PyElastica))
`FINGER`: Tendon-Driven Robot (Real-World, [Motion Caputre Device](https://optitrack.com/cameras/flex-13/))

We made available the syenthetic dataset of `ABAQUS`, `ELASTICA` and real-world dataset of `FINGER`.
The dataset can be found in `dataset` folder, also available on [IEEE DataPort](10.21227/5h7v-aq35)



