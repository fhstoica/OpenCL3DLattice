//ThreeDimLattice.cpp

//g++ ThreeDimLattice.cpp -I/usr/local/cuda-6.0/include/  -L/usr/local/cuda-6.0/lib64/ -L/usr/lib/ -lOpenCL -lstdc++ -o ThreeDimLattice

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <vector>
#include <CL/cl.h>

// Constants

const unsigned int N         = 64;
const unsigned int Steps     = 2000;

const cl_float dt            = 0.01f;
const cl_float lattice_space = 0.5f;
const cl_float damping       = 0.5f;
const cl_float vev           = 1.0f;

const unsigned int inputDepth  = N;
const unsigned int inputWidth  = N;
const unsigned int inputHeight = N;

cl_float input1[inputDepth][inputWidth][inputHeight];
cl_float input2[inputDepth][inputWidth][inputHeight];
cl_float inputVel1[inputDepth][inputWidth][inputHeight];
cl_float inputVel2[inputDepth][inputWidth][inputHeight];

const unsigned int outputDepth  = N;
const unsigned int outputWidth  = N;
const unsigned int outputHeight = N;

cl_float output1[outputDepth][outputWidth][outputHeight];
cl_float output2[outputDepth][outputWidth][outputHeight];
cl_float outputVel1[outputDepth][outputWidth][outputHeight];
cl_float outputVel2[outputDepth][outputWidth][outputHeight];

inline void errorCheck(cl_int err, const char * name){
  if(err != CL_SUCCESS){
    std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CL_CALLBACK contextCallback(const char * errInfo, const void * private_info, size_t cb, void * user_data){
  std::cout << "Error occurred during context use:"<< errInfo << std::endl;
  exit(EXIT_FAILURE);
}

void Tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " "){
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
  
  while (std::string::npos != pos || std::string::npos != lastPos){
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}

int main(int argc, char** argv){
  cl_int error_no;
  cl_uint num_of_platforms;
  cl_uint num_of_devices;
  cl_platform_id * platform_ids;
  cl_device_id * device_ids;
  cl_context context = NULL;
  cl_command_queue command_queue;
  cl_program program;
  cl_kernel kernel;
  cl_mem inputBuffer1;
  cl_mem inputBuffer2;
  cl_mem inputVelBuffer1;
  cl_mem inputVelBuffer2;
  cl_mem outputBuffer1;
  cl_mem outputBuffer2;
  cl_mem outputVelBuffer1;
  cl_mem outputVelBuffer2;
  char buffer[10240];
  
  std::vector<std::string> line_elem;
  std::string c_line;
  std::ifstream fin;
  int ix, iy, iz;

  //Read the initial data
  fin.open("initial_data.dat");
  if (fin.is_open()){
    while( fin ){
      getline(fin, c_line);
	  if(0 < c_line.size()){
		Tokenize(c_line, line_elem);
		ix = atoi(line_elem[0].c_str());
		iy = atoi(line_elem[1].c_str());
		iz = atoi(line_elem[2].c_str());
		input1[ix][iy][iz] = atof(line_elem[3].c_str());
		input2[ix][iy][iz] = atof(line_elem[4].c_str());
		inputVel1[ix][iy][iz] = 0.0f;
		inputVel2[ix][iy][iz] = 0.0f;
	  }
	  line_elem.clear();
	}
  }
  else{
	std::cerr << "Cannot open file sample_data.dat" << std::endl;
	exit(-1);
  }
  fin.close();

  //Detect the platforms
  error_no = clGetPlatformIDs(0, NULL, &num_of_platforms);
  errorCheck((error_no != CL_SUCCESS) ? error_no : (num_of_platforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs");
  
  printf("=== %d OpenCL platform(s) found: ===\n", num_of_platforms);

  platform_ids = (cl_platform_id *)alloca(sizeof(cl_platform_id) * num_of_platforms);
  error_no = clGetPlatformIDs(num_of_platforms, platform_ids, NULL);

  errorCheck((error_no != CL_SUCCESS) ? error_no : (num_of_platforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs");
  
  device_ids = NULL;
  cl_uint i = 0;

  clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 10240, buffer, NULL);
  printf("  NAME = %s\n", buffer);

  //Detect the devices
  error_no = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, NULL, &num_of_devices);
  printf("=== %d OpenCL devices: ===\n", num_of_devices);
    
  if (error_no != CL_SUCCESS && error_no != CL_DEVICE_NOT_FOUND){
    errorCheck(error_no, "clGetDeviceIDs");
  }
  else if (num_of_devices > 0){
    device_ids = (cl_device_id *)alloca(sizeof(cl_device_id) * num_of_devices);
    error_no = clGetDeviceIDs( platform_ids[i], CL_DEVICE_TYPE_GPU, num_of_devices, &device_ids[0], NULL);
    errorCheck(error_no, "clGetDeviceIDs");
  }
  
  if (device_ids == NULL) {
    std::cout << "No GPU device found" << std::endl;
    exit(-1);
  }

  //Set the context
  cl_context_properties contextProperties[] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties) platform_ids[i], 0
  };
  context = clCreateContext(contextProperties, num_of_devices, device_ids, &contextCallback, NULL, &error_no);
  errorCheck(error_no, "clCreateContext");

  std::string srcProg;
  std::ifstream kernel_in;
  
  //Read the OpenCL kernel file
  kernel_in.open("ThreeDimLattice.cl");
  if (kernel_in.is_open()){
    while( kernel_in ){
      getline(kernel_in, c_line);
	  if(0 < c_line.size()){
		srcProg += c_line + "\n";
	  }
	  line_elem.clear();
	}
  }
  else{
	std::cerr << "Cannot open file ThreeDimLattice.cl" << std::endl;
	exit(-1);
  }
  kernel_in.close();

  const char * src = srcProg.c_str();
  size_t length = srcProg.length();

  program = clCreateProgramWithSource( context, 1, &src, &length, &error_no);
  errorCheck(error_no, "clCreateProgramWithSource");

  //Build the kernel and write out the build results
  error_no = clBuildProgram(program, num_of_devices, device_ids, NULL, NULL, NULL);
  errorCheck(error_no, "clBuildProgram");
  clGetProgramBuildInfo(program, *device_ids, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
  fprintf(stderr, "CL Kernel Compilation:\n %s \n", buffer);
  
  kernel = clCreateKernel(program, "ThreeDimLattice", &error_no);
  errorCheck(error_no, "clCreateKernel");
    
  //Create the input buffers
  inputBuffer1 = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				      sizeof(cl_float) * inputDepth * inputWidth * inputHeight,
				      static_cast<void *>(input1), &error_no);
  errorCheck(error_no, "clCreateBuffer(input1)");
  
  inputBuffer2 = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				       sizeof(cl_float) * inputDepth * inputWidth * inputHeight,
				       static_cast<void *>(input2), &error_no);
  errorCheck(error_no, "clCreateBuffer(input2)");
  
  inputVelBuffer1 = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						sizeof(cl_float) * inputDepth * inputWidth * inputHeight,
						static_cast<void *>(inputVel1), &error_no);
  errorCheck(error_no, "clCreateBuffer(inputVel1)");

  inputVelBuffer2 = clCreateBuffer( context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
						sizeof(cl_float) * inputDepth * inputWidth * inputHeight,
						static_cast<void *>(inputVel2), &error_no);
  errorCheck(error_no, "clCreateBuffer(inputVel2)");
  
  //Create the output buffers
  outputBuffer1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				      sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				      NULL, &error_no);
  errorCheck(error_no, "clCreateBuffer(output1)");
    
  outputBuffer2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				       sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				       NULL, &error_no);
  errorCheck(error_no, "clCreateBuffer(output2)");

  outputVelBuffer1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
						sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
						NULL, &error_no);
  errorCheck(error_no, "clCreateBuffer(outputVel1)");

  outputVelBuffer2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
						sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
						NULL, &error_no);
  errorCheck(error_no, "clCreateBuffer(outputVel2)");
  
  //Set the command queue
  command_queue = clCreateCommandQueue(context, device_ids[0], 0, &error_no);
  errorCheck(error_no, "clCreateCommandQueue");

  //Set the kernel arguments
  error_no  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer1);
  error_no |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputBuffer2);
  error_no |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &inputVelBuffer1);
  error_no |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &inputVelBuffer2);
  error_no |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &outputBuffer1);
  error_no |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &outputBuffer2);
  error_no |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &outputVelBuffer1);
  error_no |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &outputVelBuffer2);
  error_no |= clSetKernelArg(kernel, 8, sizeof(cl_float), &dt);
  error_no |= clSetKernelArg(kernel, 9, sizeof(cl_float), &lattice_space);
  error_no |= clSetKernelArg(kernel, 10, sizeof(cl_float), &damping);
  error_no |= clSetKernelArg(kernel, 11, sizeof(cl_float), &vev);
  errorCheck(error_no, "clSetKernelArg");

  const size_t globalWorkSize[3] = {outputDepth, outputWidth, outputHeight};
  const size_t localWorkSize[3] = {4, 4, 4};
    
  //Run the kernel the required number of time steps
  for(int k = 0; k < Steps; ++k){
    error_no = clEnqueueNDRangeKernel(command_queue,
				    kernel,
				    3,
				    NULL,
				    globalWorkSize,
				    localWorkSize,
				    0,
				    NULL,
				    NULL);
    errorCheck(error_no, "clEnqueueNDRangeKernel");
    
    //Overwrite the input buffers with the results from the previous time step, and run the kernel again
    error_no = clEnqueueCopyBuffer(command_queue, outputBuffer1, inputBuffer1, 0, 0, sizeof(cl_float) * outputDepth * outputWidth * outputHeight, 0, NULL, NULL);
    errorCheck(error_no, "clEnqueueCopyBuffer - Pos 1");
    
    error_no = clEnqueueCopyBuffer(command_queue, outputBuffer2, inputBuffer2, 0, 0, sizeof(cl_float) * outputDepth * outputWidth * outputHeight, 0, NULL, NULL);
    errorCheck(error_no, "clEnqueueCopyBuffer - Pos 2");
    
    error_no = clEnqueueCopyBuffer(command_queue, outputVelBuffer1, inputVelBuffer1, 0, 0, sizeof(cl_float) * outputDepth * outputWidth * outputHeight, 0, NULL, NULL);
    errorCheck(error_no, "clEnqueueCopyBuffer - Vel 1");
    
    error_no = clEnqueueCopyBuffer(command_queue, outputVelBuffer2, inputVelBuffer2, 0, 0, sizeof(cl_float) * outputDepth * outputWidth * outputHeight, 0, NULL, NULL);
    errorCheck(error_no, "clEnqueueCopyBuffer - Vel 2");
  }
	
  //Copy the result from the GPU to the output arrays
  error_no = clEnqueueReadBuffer( command_queue, outputBuffer1, CL_TRUE, 0,
				sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				output1, 0, NULL, NULL);
  errorCheck(error_no, "clEnqueueReadBuffer - Pos 1");
  
  error_no = clEnqueueReadBuffer( command_queue, outputBuffer2, CL_TRUE, 0,
				sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				output2, 0, NULL, NULL);
  errorCheck(error_no, "clEnqueueReadBuffer - Pos 2");
  
  error_no = clEnqueueReadBuffer( command_queue, outputVelBuffer1, CL_TRUE, 0,
				sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				outputVel1, 0, NULL, NULL);
  errorCheck(error_no, "clEnqueueReadBuffer - Vel 1");
  
  error_no = clEnqueueReadBuffer( command_queue, outputVelBuffer2, CL_TRUE, 0,
				sizeof(cl_float) * outputDepth * outputWidth * outputHeight,
				outputVel2, 0, NULL, NULL);
  errorCheck(error_no, "clEnqueueReadBuffer - Vel 2");
  
  //Write the results to a file
  std::ofstream fout("output.txt");
  for (int x = 0; x < outputDepth; x++){
    for (int y = 0; y < outputWidth; y++){
      for (int z = 0; z < outputHeight; z++){
		  fout << x << " " << y << " " << z << " " << sqrt(pow(output1[x][y][z], 2) + pow(output2[x][y][z], 2)) << std::endl;
      }	
    }
  }
  fout.close();
  return(0);
}
