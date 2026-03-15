# UEGaussianSplatting (WIP)

> Still Work In Progress! Not Done Yet!

Supported Unreal Engine version: UE5.6+

## Overview
Loading and rendering highly detailed large scale assets in 3D software is always a big challenge.
Unreal Engine has the feature like Nanite for handling high poly/large scale mesh.
However, currently there isn't any solution for handling high density/large scale gaussian splats.
This project primarily aims to solve this issue though dynamic streaming methods.

## Temporary Result
- 1 million gaussian splats in 120 fps

![Temp](https://github.com/user-attachments/assets/71afa392-4e3d-49e1-ae3b-5cebe31621a8)

- Nanite-like gaussin splatting performance optimization solution

<img width="1594" height="641" alt="Nanite1" src="https://github.com/user-attachments/assets/0c028439-3bc7-4cb8-8e46-dc986a267675" />
<img width="1915" height="1025" alt="Nanite2" src="https://github.com/user-attachments/assets/71612017-103c-41b5-8240-fb13dd98f485" />



## How To Use It
- Place the plugin inside project's "Plugins" folder
- Press import button to import PLY file. A Gaussian Splat Asset will be created

(Add GIF here)

- Drag the Gaussian Splat Asset directly into the level

(Add GIF here)

- Enable/Disable Nanite through asset action if needed

(Add GIF here)


## Settings
<img width="828" height="453" alt="Image" src="https://github.com/user-attachments/assets/d7314cbf-b0b9-4748-ad5f-ba87375fbfef" />

| Category | Variable | Description |
| :--- | :--- | :--- |
| Quality | SH Order | sphere harmonics order |
| Performance | Sort Every Nth Frame | adjust gaussian splats sorting speed |
| Performance | Enable Frustum Culling | enable/disable frustum culling |
| Performance | LOD Error Threshold | LOD cluster switching sensitivity |
| Rendering | Opacity Scale | adjuct the opacity of gaussian splats|
| Rendering | Splat Scale | adjuct the scale of gaussian splats|


## Debug Console Command
- `gs.ShowClusterBounds 1`: enable Nanite cluster preview (set to 0 to disable preview)
- `gs.DebugForceLODLevel ?`: force render a specific LOD cluster for debugging purpose (can be 1,2,3,4...)
- `gs.MaxRenderBudget ?`: Limit the max number of visible splats(after culling) for saving VRAM. By default there is no limitaion(0). Set the max cap to decrease the VRAM usage (ex:3,000,000). The culling will start from the splats that is far from the camera


