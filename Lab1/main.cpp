#include <iostream>
#include <iomanip>
#include <fstream>
#include <CL/cl.hpp>

cl::Platform get_platform() {
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0){
		std::cout << " No platforms found. Check OpenCL installation!" << std::endl;
		exit(1);
	}
	cl::Platform platform = all_platforms[0];
	std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
	return platform;
}

cl::Device get_device(cl::Platform& platform) {
	std::vector<cl::Device> all_devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0){
		std::cout << " No devices found. Check OpenCL installation!" << std::endl;
		exit(1);
	}
	cl::Device device = all_devices[1];
	std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	return device;
}

cl::Program get_program(cl::Device& device, cl::Context& context, std::string const & program_file_name) {
	std::ifstream kernel_file(program_file_name);
	std::string kernel_code(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, { kernel_code.c_str(), kernel_code.length() });

	cl::Program program(context, sources);
	if (program.build({ device }) != CL_SUCCESS){
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		exit(1);
	}
	return program;
}

void readMatrix(std::ifstream& in, std::vector<float>& matrix, size_t size) {
	for (size_t i = 0; i < size; ++i)
		for (size_t j = 0; j < size; ++j)
			in >> matrix[i * size + j];
}

void writeMatrix(std::ofstream& out, std::vector<float> const & matrix, size_t size) {
	out << std::fixed << std::setprecision(3);

	for (size_t i = 0; i < size; ++i) {
		for (size_t j = 0; j < size; ++j)
			out << matrix[i * size + j] << " ";
		out << std::endl;
	}
}

void main_program(cl::Device& device, cl::Context& context, cl::Program& program) {
	size_t a_size = 0;
	size_t b_size = 0;

	std::ifstream in("input.txt");
	std::ofstream out("output.txt");
	in >> a_size >> b_size;

	std::vector<float> A(a_size * a_size, 0.0);
	std::vector<float> B(b_size * b_size, 0.0);
	std::vector<float> C(a_size * a_size, 0.0);

	readMatrix(in, A, a_size);
	readMatrix(in, B, b_size);

	size_t const block_size = 16;

	cl::CommandQueue queue(context, device);

	cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, sizeof(float) * A.size());
	cl::Buffer buffer_B(context, CL_MEM_READ_ONLY, sizeof(float) * B.size());
	cl::Buffer buffer_C(context, CL_MEM_WRITE_ONLY, sizeof(float) * A.size());

	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * A.size(), &A[0]);
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(float) * B.size(), &B[0]);
	queue.finish();

	size_t global_size = ((a_size + block_size - 1) / block_size) * block_size;

	cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, int, int> convolution(cl::Kernel(program, "convolution"));
	cl::EnqueueArgs eargs(queue, cl::NullRange, cl::NDRange(global_size, global_size), cl::NDRange(block_size, block_size));
	convolution(eargs, buffer_A, buffer_B, buffer_C, a_size, b_size).wait();

	queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * A.size(), &C[0]);

	writeMatrix(out, C, a_size);
}

int main()
{
	try {
		cl::Platform default_platform = get_platform();
		cl::Device default_device = get_device(default_platform);
		cl::Context context({ default_device });
		cl::Program program = get_program(default_device, context, "convolution.cl");

		main_program(default_device, context, program);
	} catch (std::exception &e) {
		std::cout << std::endl << e.what() <<  std::endl;
	}
	

	return 0;
}