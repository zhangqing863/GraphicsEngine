#ifndef QZRT_CORE_BVH_H
#define QZRT_CORE_BVH_H

#include "../core/shape.h"
//#define BVH_DEBUG_INFO

#define STACKSIZE 20000

namespace raytracer {

	__device__ inline int BoxCompareOnAxisX(const void* a, const void* b);
	__device__ inline int BoxCompareOnAxisY(const void* a, const void* b);
	__device__ inline int BoxCompareOnAxisZ(const void* a, const void* b);



	class BVHNode :public Shape {
	public:
		/*Shape* left;
		Shape* right;*/
		//Bounds3f box;
		//int flag; //flag=0左子树没有设置；flag=1左子树设置，右子树没有设置；flag=2出栈条件
		//int numShapes;

		Shape** nodes = nullptr;
		Shape** shapes = nullptr;

		__device__ BVHNode() {}
		__device__ BVHNode(Shape** shapes, int n, Shape** nodes, Float time0 = 0.f, Float time1 = 1.f);

		__device__ virtual bool Hit(const Ray& ray, HitRecord& rec)const override;

		// 通过 Shape 继承
		__device__ virtual bool BoundingBox(Bounds3f& box) const override;


	};

	__device__ inline BVHNode::BVHNode(Shape** shapes, int n, Shape** nodes, Float time0, Float time1) {
		numShapes = n;
		flag = 0;
		this->nodes = nodes;
		this->shapes = shapes;
	}

	__device__ inline bool BVHNode::Hit(const Ray& ray, HitRecord& rec) const {
		//printf("Hit BVHNode!\n");
		// 栈
		//thrust::device_vector<int> stack;
		int stack[20];
		int sp = 0;
		stack[sp++] = 0;
		//stack.push_back(sp);
		bool isHit = false;
		rec.t = Infinity;
		while (sp > 0) {
			//printf("Hit BVHNode! sp: %d\n", sp);
			int top = stack[--sp];
			//int top = stack.back();
			//stack.pop_back();
			Shape* node = nodes[top];

			if (node->box.IntersectP(ray)) { // 如果击中了box
				// 是叶子节点，直接调用Shape的击中方法
				if (node->flag == 3) { 
					int L = node->left;
					int R = node->right;
					HitRecord leftRec, rightRec;
					bool hitLeft = shapes[L]->Hit(ray, leftRec);
					bool hitRight = shapes[R]->Hit(ray, rightRec);
					/* 这里击中了叶节点也不能直接返回，否则击中的结果是错误的
					* 由于采用的是栈，而不是递归，因此很多设计会更难一些，
					* 当我们击中了当前盒子的某个shape时，可能在另一个盒子中可以击中更近的一个shape，但是由于构建BVH的时候，
					* 可能后者被分到了一个小盒子里，导致我们在与盒子判断求交时认为大盒子我们更先击中，于是忽略掉那个小盒子，导致更近的那个shape被忽略掉
					* 因此除非有盒子没被击中可以直接忽略掉，其它都需要进行子节点求交，最后得到最近的shape
					*/ 
					if (hitLeft && hitRight) {
						if (leftRec.t < rightRec.t && leftRec.t < rec.t) {
							rec = leftRec;
						}
						else if(leftRec.t >= rightRec.t && rightRec.t < rec.t) {
							rec = rightRec;
						}
						isHit = true;
					}
					else if (hitLeft) {
						if (leftRec.t < rec.t) {
							rec = leftRec;
						}
						isHit = true;
					}
					else if (hitRight) {
						if (rightRec.t < rec.t) {
							rec = rightRec;
						}
						isHit = true;
					}
				}
				else {
					bool leftHit = nodes[node->left]->box.IntersectP(ray);
					bool rightHit = nodes[node->right]->box.IntersectP(ray);
					if (leftHit && rightHit) {
						stack[sp++] = node->right;
						stack[sp++] = node->left;
						//stack.push_back(node->right);
						//stack.push_back(node->left);
					}
					else if (leftHit) {
						stack[sp++] = node->left;
						//stack.push_back(node->left);
					}
					else if (rightHit) {
						stack[sp++] = node->right;
						//stack.push_back(node->right);
					}
				}
			}
			
		}
		if (isHit) return true;
		return false;
	}

	__device__ inline bool BVHNode::BoundingBox(Bounds3f& box) const {
		box = this->box;
		return true;
	}

	__device__ inline int BoxCompareOnAxisX(const void* a, const void* b) {
		//printf("Stacking BoxCompareOnAxisX...\n");
		Bounds3f boxLeft, boxRight;
		Shape* aShapes = *(Shape**)a;
		Shape* bShapes = *(Shape**)b;
		if (!aShapes->BoundingBox(boxLeft) || !bShapes->BoundingBox(boxRight)) {
			boxLeft = Bounds3f();
			boxRight = Bounds3f();
		}
		if (boxLeft.pMin.x - boxRight.pMin.x < 0.f) {
			return -1;
		}
		else {
			return 1;
		}
	}

	__device__ inline int BoxCompareOnAxisY(const void* a, const void* b) {
		//printf("Stacking BoxCompareOnAxisY...\n");
		Bounds3f boxLeft, boxRight;
		Shape* aShapes = *(Shape**)a;
		Shape* bShapes = *(Shape**)b;
		if (!aShapes->BoundingBox(boxLeft) || !bShapes->BoundingBox(boxRight)) {
			boxLeft = Bounds3f();
			boxRight = Bounds3f();
		}
		if (boxLeft.pMin.y - boxRight.pMin.y < 0.f) {
			return -1;
		}
		else {
			return 1;
		}
	}

	__device__ inline int BoxCompareOnAxisZ(const void* a, const void* b) {
		//printf("Stacking BoxCompareOnAxisZ...\n");
		Bounds3f boxLeft, boxRight;
		Shape* aShapes = *(Shape**)a;
		Shape* bShapes = *(Shape**)b;
		if (!aShapes->BoundingBox(boxLeft) || !bShapes->BoundingBox(boxRight)) {
			boxLeft = Bounds3f();
			boxRight = Bounds3f();
		}
		if (boxLeft.pMin.z - boxRight.pMin.z < 0.f) {
			return -1;
		}
		else {
			return 1;
		}
	}

	/*__device__ inline Shape* create_temp_world(Shape** shapes, int n) {
		return new ShapeList(shapes, n);
	}*/

	__device__ inline Shape* CreateBVHNode(Shape** shapes, int n, Shape** nodes, curandState* local_rand_state, Float time0, Float time1) {
		// flag={-1,0,1,2}; 
		// -1(表示普通的Shape，没有左右孩子)
		// 0(表示BVHNode，且左右孩子为空)
		// 1(表示BVHNode，只有左孩子)
		// 2(表示BVHNode，只有右孩子)
		Shape** temp = shapes;
		Shape** stack = new Shape * [n]; // 定义栈
		int* numShapesBeginStack = new int[n]; // 定义栈
		int* numShapesEndStack = new int[n]; // 定义栈
		int top = -1;
		int size = -1;
		BVHNode* root = new BVHNode(temp, n, nodes, time0, time1);
		nodes[++size] = root;
		//root->flag = 0;
		stack[++top] = root;
		numShapesBeginStack[top] = 0;
		numShapesEndStack[top] = n - 1;
		// 按照坐标轴的值排序
		int axis = int(3 * curand_uniform(local_rand_state));
		if (axis == 0) {
			Qsort(temp, n, sizeof(Shape*), BoxCompareOnAxisX);
		}
		else if (axis == 1) {
			Qsort(temp, n, sizeof(Shape*), BoxCompareOnAxisY);
		}
		else {
			Qsort(temp, n, sizeof(Shape*), BoxCompareOnAxisZ);
		}
#ifdef BVH_DEBUG_INFO
		printf("Stacking...\n");
#endif // BVH_DEBUG_INFO
		int progress = 0; // 记录进度
		int percentPg = 10; // 间隔时间
		int status = 1;
		while (top != -1) {
			progress++;
			if (progress % percentPg == 0) {
				if (status == 1) {
					printf("\rBuilding BVH.");
					status = 2;
				}
				else if (status == 2) {
					printf("\rBuilding BVH..");
					status = 3;
				}
				else {
					printf("\rBuilding BVH..."); 
					status = 1;
				}
			}
			
#ifdef BVH_DEBUG_INFO
			printf("Top:%d, ", top);
#endif // BVH_DEBUG_INFO
			int tempN = numShapesEndStack[top] - numShapesBeginStack[top] + 1;
			int tempBegin = numShapesBeginStack[top];
			int tempEnd = numShapesEndStack[top];
#ifdef BVH_DEBUG_INFO
			printf("begin:%d, end:%d, ", tempBegin, tempEnd);
#endif // BVH_DEBUG_INFO
			// 按照坐标轴的值排序

			if (tempN == 1) { // 叶子节点
				stack[top]->left = tempBegin;
				stack[top]->right = tempBegin;
				stack[top]->flag = 3;
				//temp[tempBegin]->flag = -1;
#ifdef BVH_DEBUG_INFO
				printf("left[begin:%d, end:%d], right[begin:%d, end:%d], ", tempBegin, tempBegin, tempBegin, tempBegin);
				printf("stack leaf node(n==1)\n");
#endif // BVH_DEBUG_INFO
			}
			else if (tempN == 2) { // 叶子节点
				stack[top]->left = tempBegin;
				stack[top]->right = tempBegin + 1;
				stack[top]->flag = 3;
				//temp[tempBegin]->flag = -1;
				//temp[tempBegin + 1]->flag = -1;
#ifdef BVH_DEBUG_INFO
				printf("left[begin:%d, end:%d], right[begin:%d, end:%d], ", tempBegin, tempBegin, tempBegin + 1, tempBegin + 1);
				printf("stack leaf node(n==2)\n");
#endif // BVH_DEBUG_INFO
			}
			else { // 中间节点
				
				if (stack[top]->flag == 0) {
					BVHNode* node = new BVHNode(temp + tempBegin, tempN / 2, nodes, time0, time1);
					nodes[++size] = node;
					stack[top]->left = size;
					stack[top]->flag = 1;
					stack[++top] = node;
					numShapesBeginStack[top] = tempBegin;
					numShapesEndStack[top] = tempBegin + tempN / 2 - 1;
#ifdef BVH_DEBUG_INFO
					printf("left[begin:%d, end:%d], ", tempBegin, tempBegin + tempN / 2 - 1);
					printf("stack left node\n");
#endif // BVH_DEBUG_INFO
					axis = int(3 * curand_uniform(local_rand_state));
					if (axis == 0) {
						Qsort(temp + numShapesBeginStack[top], numShapesEndStack[top] - numShapesBeginStack[top] + 1, sizeof(Shape*), BoxCompareOnAxisX);
					}
					else if (axis == 1) {
						Qsort(temp + numShapesBeginStack[top], numShapesEndStack[top] - numShapesBeginStack[top] + 1, sizeof(Shape*), BoxCompareOnAxisY);
					}
					else {
						Qsort(temp + numShapesBeginStack[top], numShapesEndStack[top] - numShapesBeginStack[top] + 1, sizeof(Shape*), BoxCompareOnAxisZ);
					}
				}
				else if (stack[top]->flag == 1) {
					BVHNode* node = new BVHNode(temp + tempBegin + tempN / 2, tempN - tempN / 2, nodes, time0, time1);
					nodes[++size] = node;
					stack[top]->right = size;
					stack[top]->flag = 2;
					stack[++top] = node;
					numShapesBeginStack[top] = tempBegin + tempN / 2;
					numShapesEndStack[top] = tempEnd;
#ifdef BVH_DEBUG_INFO
					printf("right[begin:%d, end:%d], ", tempBegin + tempN / 2, tempEnd);
					printf("stack right node\n");
#endif // BVH_DEBUG_INFO
				}
			}
			while (top >= 0 && stack[top]->flag >= 2) {
				Bounds3f leftBox, rightBox;
				// 叶节点
				if (stack[top]->flag == 3 && stack[top]->left >= 0 && stack[top]->right >= 0) {
					if (shapes[stack[top]->left]->BoundingBox(leftBox) && shapes[stack[top]->right]->BoundingBox(rightBox)) {
						stack[top]->box = Union(leftBox, rightBox);
					}
					else {
						stack[top]->box = Bounds3f();
					}
				}
				else if (stack[top]->flag == 2 && stack[top]->left >= 0 && stack[top]->right >= 0) {
					if (nodes[stack[top]->left]->BoundingBox(leftBox) && nodes[stack[top]->right]->BoundingBox(rightBox)) {
						stack[top]->box = Union(leftBox, rightBox);
					}
					else {
						stack[top]->box = Bounds3f();
					}
				}
				else {
					stack[top]->box = Bounds3f();
#ifdef BVH_DEBUG_INFO
					printf("The Bounding Box is max\n");
#endif // BVH_DEBUG_INFO
				}
#ifdef BVH_DEBUG_INFO
				Bounds3f tempBox;
				stack[top]->BoundingBox(tempBox);
				printf("-------------------------------------------------\n");
				printf("The %dth Shape Bounding Box is:[%f, %f, %f]\n", top, stack[top]->box.pMin.x, stack[top]->box.pMin.y, stack[top]->box.pMin.z);
				printf("The %dth Shape Left Bounding Box is:[%f, %f, %f]\n", top, tempBox.pMin.x, tempBox.pMin.y, tempBox.pMin.z);
				printf("The %dth Shape Left Index is:[%d]\n", top, stack[top]->left);
				printf("The %dth Shape Right Index is:[%d]\n", top, stack[top]->right);
				printf("-------------------------------------------------\n");
#endif // BVH_DEBUG_INFO

				top--;
			}
			
		}

#ifdef BVH_DEBUG_INFO
		// 后序遍历
		printf("stack complete! top:%d\n", top);
		stack[++top] = root;
		Shape* lastPop = nullptr;
		int i = 0;
		while (top > -1) {
			while (stack[top]->left) {
				stack[++top] = nodes[stack[top]->left];
			}
			while (top > -1) {
				if (lastPop == nodes[stack[top]->right] || stack[top]->right > 0) {
					Bounds3f box;
					stack[top]->BoundingBox(box);
					printf("The %dth Shape Bounding Box is:[%f, %f, %f] and have boys:%d\n", top + 1, box.pMin.x, box.pMin.y, box.pMin.z, ((BVHNode*)stack[top])->numShapes);
					i++;
					lastPop = stack[top];
					top--;
				}
				else if (stack[top]->right > 0) {
					stack[++top] = nodes[stack[top]->right];
					break;
				}
			}
		}
		printf("Count %d\n", i);
#endif // BVH_DEBUG_INFO
#ifdef BVH_DEBUG_INFO
		for (int i = 0; i < size; i++) {
			Bounds3f box;
			nodes[i]->BoundingBox(box);
			printf("--------------------------------------------------------------------\n");
			printf("The %dth Shape Flag is:[ %d ]\n", i + 1, nodes[i]->flag);
			printf("The %dth Shape Bounding Box is:[%f, %f, %f] and [%f, %f, %f]\n", i + 1, box.pMin.x, box.pMin.y, box.pMin.z, box.pMax.x, box.pMax.y, box.pMax.z);
			printf("The %dth Shape Left and Right node index is:[ %d ] and [ %d ]\n", i + 1, nodes[i]->left, nodes[i]->right);
		}
#endif // BVH_DEBUG_INFO
		root->numNodes = size;
		delete* stack;
		delete[]numShapesBeginStack;
		delete[]numShapesEndStack;

		printf("\rBuilt BVH    :)\n");
		return root;
	}




	//__device__ inline bool DeleteBVHNode(Shape** nodes, int numNodes) {
	//	for (int i = 0; i < numNodes; i++) {
	//		printf("%d\n", i);
	//		printf("nodes left %d and right %d and flag %d \n",((BVHNode*)nodes[i])->left, ((BVHNode*)nodes[i])->right, ((BVHNode*)nodes[i])->flag );
	//		delete (BVHNode*)nodes[i];
	//	}
	//	printf("delete nodes");
	//}


//	__device__ inline bool CreateBVHNode(Shape** shapes, int n, Shape** nodes, curandState* local_rand_state, Float time0, Float time1) {
//		// flag={-1,0,1,2}; 
//		// -1(表示普通的Shape，没有左右孩子)
//		// 0(表示BVHNode，且左右孩子为空)
//		// 1(表示BVHNode，只有左孩子)
//		// 2(表示BVHNode，只有右孩子)
//		Shape** temp = shapes;
//		Shape** stack = new Shape * [n]; // 定义栈
//		int* numShapesBeginStack = new int[n]; // 定义栈
//		int* numShapesEndStack = new int[n]; // 定义栈
//		int top = -1;
//		int size = -1;
//		BVHNode* root = (BVHNode*)nodes[0];
//		nodes[++size] = root;
//		//root->flag = 0;
//		stack[++top] = root;
//		numShapesBeginStack[top] = 0;
//		numShapesEndStack[top] = n - 1;
//		// 按照坐标轴的值排序
//		int axis = int(3 * curand_uniform(local_rand_state));
//		if (axis == 0) {
//			Qsort(temp, n, sizeof(Shape*), BoxCompareOnAxisX);
//		}
//		else if (axis == 1) {
//			Qsort(temp, n, sizeof(Shape*), BoxCompareOnAxisY);
//		}
//		else {
//			Qsort(temp, n, sizeof(Shape*), BoxCompareOnAxisZ);
//		}
//#ifdef BVH_DEBUG_INFO
//		printf("Stacking...\n");
//#endif // BVH_DEBUG_INFO
//		while (top != -1) {
//#ifdef BVH_DEBUG_INFO
//			printf("Top:%d, ", top);
//#endif // BVH_DEBUG_INFO
//			int tempN = numShapesEndStack[top] - numShapesBeginStack[top] + 1;
//			int tempBegin = numShapesBeginStack[top];
//			int tempEnd = numShapesEndStack[top];
//#ifdef BVH_DEBUG_INFO
//			printf("begin:%d, end:%d, ", tempBegin, tempEnd);
//#endif // BVH_DEBUG_INFO
//			// 按照坐标轴的值排序
//
//			if (tempN == 1) { // 叶子节点
//				stack[top]->left = tempBegin;
//				stack[top]->right = tempBegin;
//				stack[top]->flag = 3;
//				//temp[tempBegin]->flag = -1;
//#ifdef BVH_DEBUG_INFO
//				printf("left[begin:%d, end:%d], right[begin:%d, end:%d], ", tempBegin, tempBegin, tempBegin, tempBegin);
//				printf("stack leaf node(n==1)\n");
//#endif // BVH_DEBUG_INFO
//			}
//			else if (tempN == 2) { // 叶子节点
//				stack[top]->left = tempBegin;
//				stack[top]->right = tempBegin + 1;
//				stack[top]->flag = 3;
//				//temp[tempBegin]->flag = -1;
//				//temp[tempBegin + 1]->flag = -1;
//#ifdef BVH_DEBUG_INFO
//				printf("left[begin:%d, end:%d], right[begin:%d, end:%d], ", tempBegin, tempBegin, tempBegin + 1, tempBegin + 1);
//				printf("stack leaf node(n==2)\n");
//#endif // BVH_DEBUG_INFO
//			}
//			else { // 中间节点
//
//				if (stack[top]->flag == 0) {
//					BVHNode* node = (BVHNode*)nodes[++size];
//					stack[top]->left = size;
//					stack[top]->flag = 1;
//					stack[++top] = node;
//					numShapesBeginStack[top] = tempBegin;
//					numShapesEndStack[top] = tempBegin + tempN / 2 - 1;
//#ifdef BVH_DEBUG_INFO
//					printf("left[begin:%d, end:%d], ", tempBegin, tempBegin + tempN / 2 - 1);
//					printf("stack left node\n");
//#endif // BVH_DEBUG_INFO
//					axis = int(3 * curand_uniform(local_rand_state));
//					if (axis == 0) {
//						Qsort(temp + numShapesBeginStack[top], numShapesEndStack[top] - numShapesBeginStack[top] + 1, sizeof(Shape*), BoxCompareOnAxisX);
//					}
//					else if (axis == 1) {
//						Qsort(temp + numShapesBeginStack[top], numShapesEndStack[top] - numShapesBeginStack[top] + 1, sizeof(Shape*), BoxCompareOnAxisY);
//					}
//					else {
//						Qsort(temp + numShapesBeginStack[top], numShapesEndStack[top] - numShapesBeginStack[top] + 1, sizeof(Shape*), BoxCompareOnAxisZ);
//					}
//				}
//				else if (stack[top]->flag == 1) {
//					BVHNode* node = (BVHNode*)nodes[++size];
//					nodes[++size] = node;
//					stack[top]->right = size;
//					stack[top]->flag = 2;
//					stack[++top] = node;
//					numShapesBeginStack[top] = tempBegin + tempN / 2;
//					numShapesEndStack[top] = tempEnd;
//#ifdef BVH_DEBUG_INFO
//					printf("right[begin:%d, end:%d], ", tempBegin + tempN / 2, tempEnd);
//					printf("stack right node\n");
//#endif // BVH_DEBUG_INFO
//				}
//			}
//			while (top >= 0 && stack[top]->flag >= 2) {
//				Bounds3f leftBox, rightBox;
//				// 叶节点
//				if (stack[top]->flag == 3 && stack[top]->left >= 0 && stack[top]->right >= 0) {
//					if (shapes[stack[top]->left]->BoundingBox(leftBox) && shapes[stack[top]->right]->BoundingBox(rightBox)) {
//						stack[top]->box = Union(leftBox, rightBox);
//					}
//					else {
//						stack[top]->box = Bounds3f();
//					}
//				}
//				else if (stack[top]->flag == 2 && stack[top]->left >= 0 && stack[top]->right >= 0) {
//					if (nodes[stack[top]->left]->BoundingBox(leftBox) && nodes[stack[top]->right]->BoundingBox(rightBox)) {
//						stack[top]->box = Union(leftBox, rightBox);
//					}
//					else {
//						stack[top]->box = Bounds3f();
//					}
//				}
//				else {
//					stack[top]->box = Bounds3f();
//#ifdef BVH_DEBUG_INFO
//					printf("The Bounding Box is max\n");
//#endif // BVH_DEBUG_INFO
//				}
//#ifdef BVH_DEBUG_INFO
//				Bounds3f tempBox;
//				stack[top]->BoundingBox(tempBox);
//				printf("-------------------------------------------------\n");
//				printf("The %dth Shape Bounding Box is:[%f, %f, %f]\n", top, stack[top]->box.pMin.x, stack[top]->box.pMin.y, stack[top]->box.pMin.z);
//				printf("The %dth Shape Left Bounding Box is:[%f, %f, %f]\n", top, tempBox.pMin.x, tempBox.pMin.y, tempBox.pMin.z);
//				printf("The %dth Shape Left Index is:[%d]\n", top, stack[top]->left);
//				printf("The %dth Shape Right Index is:[%d]\n", top, stack[top]->right);
//				printf("-------------------------------------------------\n");
//#endif // BVH_DEBUG_INFO
//
//				top--;
//			}
//
//		}
//
//#ifdef BVH_DEBUG_INFO
//		// 后序遍历
//		printf("stack complete! top:%d\n", top);
//		stack[++top] = root;
//		Shape* lastPop = nullptr;
//		int i = 0;
//		while (top > -1) {
//			while (stack[top]->left) {
//				stack[++top] = nodes[stack[top]->left];
//			}
//			while (top > -1) {
//				if (lastPop == nodes[stack[top]->right] || stack[top]->right > 0) {
//					Bounds3f box;
//					stack[top]->BoundingBox(box);
//					printf("The %dth Shape Bounding Box is:[%f, %f, %f] and have boys:%d\n", top + 1, box.pMin.x, box.pMin.y, box.pMin.z, ((BVHNode*)stack[top])->numShapes);
//					i++;
//					lastPop = stack[top];
//					top--;
//				}
//				else if (stack[top]->right > 0) {
//					stack[++top] = nodes[stack[top]->right];
//					break;
//				}
//			}
//		}
//		printf("Count %d\n", i);
//#endif // BVH_DEBUG_INFO
//#ifdef BVH_DEBUG_INFO
//		for (int i = 0; i < size; i++) {
//			Bounds3f box;
//			nodes[i]->BoundingBox(box);
//			printf("--------------------------------------------------------------------\n");
//			printf("The %dth Shape Flag is:[ %d ]\n", i + 1, nodes[i]->flag);
//			printf("The %dth Shape Bounding Box is:[%f, %f, %f] and [%f, %f, %f]\n", i + 1, box.pMin.x, box.pMin.y, box.pMin.z, box.pMax.x, box.pMax.y, box.pMax.z);
//			printf("The %dth Shape Left and Right node index is:[ %d ] and [ %d ]\n", i + 1, nodes[i]->left, nodes[i]->right);
//		}
//#endif // BVH_DEBUG_INFO
//		root->numNodes = size;
//		delete* stack;
//		delete[]numShapesBeginStack;
//		delete[]numShapesEndStack;
//
//		return true;
//	}
}



#endif // QZRT_CORE_BVH_H

