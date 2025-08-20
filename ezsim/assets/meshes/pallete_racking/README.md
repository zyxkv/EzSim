# pallete racking assets

## w/o top face (means its top face is not covered, robot/drone/vehicle cannot walk/land/move on top)
- wall_1.glb size:[0.2,2,2]
- wall_2.glb size:[0.2,2,4]
note: wall are basic side support wall of all following objects. 
      wall_1 is similar with square gate but with one diagonal bar strengthify itself.
      wall_2 is two wall_1 stack at z-axis but wall_2 has opposite diagnoal bar in each sqaure gate compared with wall_1 

- frame_1.glb size:[4,2,2]
- frame_2.glb size:[4,2,4]
note： frame_1 use two x-direction bar left/right to connect two wall_1 to compose a 4m * 2m * 2m frame
       frame_2 use two frame_1 stack at z-axis, but has opposite diagnoal bar with frame_1 like wall_2 versus wall_1


## w topface (means its top face is covered, robot/drone/vehicle can walk/land/move on top)
- cube_1.glb size:[2,2,2] has top face, x/y plane can pass
- cube_2.glb size:[2,2,4] has top face, x/y plane can pass
note: cube_1 use two x-direction bar left/right to connect two wall_1 to compose a 2m * 2m * 2m frame and add a topface on it 
      cube_2 use two cube_1 stack at z-axis, but has opposite diagnoal bar with frame_1 like wall_2 versus wall_1, no intermidiate x-y plane only has top-face


- stage_1.glb size:[4,2,2] has top face , x-direction can pass
- stage_2.glb size:[4,2,4] has top face , x-direction can pass 
note: stage_1 is based on frame_1 and add a topface on it 
      stage_2 is based on frame_2 and add a topface on it 

- stair_1.glb size:[unmeasured,2,2] has top face, x-direction can pass
- stair_2.glb size:[unmeasured,2,4] has top face, x-direction can pass

note: stair_1 is cube_1 attach with a x-direction slope from ground to it, means it can make a slope unit in a bigger map 
      stair_2 is cube_1 connect to cube_2 with a x-positive direction slope, means it can make a higher slope unit in a bigger map 
      stair x-axis size is not measured for now, load it and use some aabb technique can get the accurate size.



note: 
1. all these asset is x-poitive-direction aligned, initial pos 0,0,0 make all of them touch ground
2. rotate and place them correctly can build complex 3d route for robot learning dojo. (cube,stage,stairs can build map for robot/vehicle/quadruples etc)
3. cube,stage,stairs and wall_1,wall_2,frame_1,frame_2 can all be used in build drone fly indoor dojo. small size drone can fly through assets' x-direction because these glb assets only has a diagnol edge .


    