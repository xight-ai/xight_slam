Header header

float32[3] self_v
float32[6] transform_between

int32 saveFrame
float32[6] transform_map

sensor_msgs/PointCloud2 cloud_org

sensor_msgs/PointCloud2 cloud_static
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature

sensor_msgs/PointCloud2 cloud_moving
sensor_msgs/PointCloud2 cloud_bounding

# *******************************************************************************************************************
int32[] startRingIndex
int32[] endRingIndex

int32[]  pointColInd # point column index in range image
float32[] pointRange # point range 

int64 imuAvailable
int64 odomAvailable

# Attitude for LOAM initialization
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Initial guess from imu pre-integration
float32 initialGuessX
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

sensor_msgs/PointCloud2 cloud_deskewed