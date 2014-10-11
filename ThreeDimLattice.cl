__kernel void ThreeDimLattice(	const __global float * const pos_input1,
								const __global float * const pos_input2,
								const __global float * const vel_input1,
								const __global float * const vel_input2,
								__global float * const pos_output1,
								__global float * const pos_output2,
								__global float * const vel_output1,	
								__global float * const vel_output2,	
								const float dt, 
								const float a,
								const float damping,
								const float vev)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  int x_p;
  int y_p;
  int z_p;

  int x_m;
  int y_m;
  int z_m;

  const int max_x = get_global_size(0);
  const int max_y = get_global_size(1);
  const int max_z = get_global_size(2);
  
  float temp1;
  float temp2;

  //Calculate the index for the linearized 3D array
  const int idx = x * max_y * max_z + y * max_z + z;
 
  //Set periodic boundary conditions
  if(x - 1 < 0){ x_m = max_x - 1; }
  else{ x_m = x - 1; }

  if(y - 1 < 0){ y_m = max_y - 1; }
  else{ y_m = y - 1; }

  if(z - 1 < 0){ z_m = max_z - 1; }
  else{ z_m = z - 1; }

  if(max_x <= x + 1){ x_p = 0; }
  else{ x_p = x + 1; }

  if(max_y <= y + 1){ y_p = 0; }
  else{ y_p = y + 1; }

  if(max_z <= z + 1){ z_p = 0; }
  else{ z_p = z + 1; }

  //Calculate the indices for positions +/- along each axis for linearized 3D array
  const int idx_x_pyz = x_p * max_y * max_z + y * max_z + z;
  const int idx_x_myz = x_m * max_y * max_z + y * max_z + z;

  const int idx_xy_pz = x * max_y * max_z + y_p * max_z + z;
  const int idx_xy_mz = x * max_y * max_z + y_m * max_z + z;

  const int idx_xyz_p = x * max_y * max_z + y * max_z + z_p;
  const int idx_xyz_m = x * max_y * max_z + y * max_z + z_m;

  //Calculate force from the potential term, real part
  temp1 = -pos_input1[idx]*(pow(pos_input1[idx], 2) + pow(pos_input2[idx], 2) - vev);

  //Calculate force from interaction with neighbors, real part
  temp1 += (pos_input1[idx_x_pyz] - 2 * pos_input1[idx] + pos_input1[idx_x_myz])/(2 * a);
  temp1 += (pos_input1[idx_xy_pz] - 2 * pos_input1[idx] + pos_input1[idx_xy_mz])/(2 * a);
  temp1 += (pos_input1[idx_xyz_p] - 2 * pos_input1[idx] + pos_input1[idx_xyz_m])/(2 * a);

  //Updated the velocity term, real part
  vel_output1[idx] = vel_input1[idx] * (1 - damping * dt) + temp1 * dt;

  //Update the position term, real part
  pos_output1[idx] = pos_input1[idx] + vel_output1[idx] * dt;

  //Calculate force from the potential term, imaginary part
  temp2 = -pos_input2[idx]*(pow(pos_input1[idx], 2) + pow(pos_input2[idx], 2) - vev);

  //Calculate force from interaction with neighbors, imaginary part
  temp2 += (pos_input2[idx_x_pyz] - 2 * pos_input2[idx] + pos_input2[idx_x_myz])/(2 * a);
  temp2 += (pos_input2[idx_xy_pz] - 2 * pos_input2[idx] + pos_input2[idx_xy_mz])/(2 * a);
  temp2 += (pos_input2[idx_xyz_p] - 2 * pos_input2[idx] + pos_input2[idx_xyz_m])/(2 * a);

  //Update the velocity term, imaginary part
  vel_output2[idx] = vel_input2[idx] * (1 - damping * dt) + temp2 * dt;
  
  //Update the position term, imaginary part
  pos_output2[idx] = pos_input2[idx] + vel_output2[idx] * dt;
}
