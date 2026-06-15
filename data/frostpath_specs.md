# Frostpath F2 — Technical Specifications

The Frostpath F2 is Aurelia Robotics' second-generation cold-storage picking
robot, released in 2024.

Locomotion: four-wheel differential drive with heated polymer treads rated for
icy floors. Maximum speed is 1.8 meters per second when loaded.

Power: a 2.1 kWh lithium-iron-phosphate battery pack with an integrated thermal
jacket. Continuous operating time is about 6 hours at -20 Celsius. The battery
uses a self-warming pre-charge cycle that draws heat before drawing current.

Sensing: the F2 carries a forward stereo depth camera, a 16-line spinning lidar,
and anti-fog heated lens covers. Localization uses a fused lidar-inertial SLAM
stack running on an onboard NVIDIA Jetson Orin module.

Payload: the F2 can lift tote boxes up to 18 kilograms and stack them three high
on its internal rack. It is designed for aisles at least 1.4 meters wide.
