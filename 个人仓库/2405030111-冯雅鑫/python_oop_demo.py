# Python面向对象核心：类的定义、实例化、继承
class AIStudy:
    # 初始化属性
    def __init__(self, name, course):
        self.name = name
        self.course = course

    # 定义方法
    def learn(self):
        return f"{self.name}正在学习{self.course}课程"

# 子类继承父类
class DeepLearningStudy(AIStudy):
    def __init__(self, name, course, framework):
        super().__init__(name, course)
        self.framework = framework

    def learn_framework(self):
        return f"{self.name}使用{self.framework}实现{self.course}"

# 异常处理（try-except）
def calculate_tensor_dim(a, b):
    try:
        res = a / b
        return f"张量运算结果：{res}"
    except ZeroDivisionError:
        return "异常：除数不能为0，符合Tensor运算维度要求"
    except TypeError:
        return "异常：输入类型错误，需为数值型张量"

# 代码运行测试
if __name__ == "__main__":
    # 面向对象测试
    stu = DeepLearningStudy("AI学员", "PyTorch Tensor", "PyTorch")
    print(stu.learn())
    print(stu.learn_framework())
    # 异常处理测试
    print(calculate_tensor_dim(10, 2))
    print(calculate_tensor_dim(8, 0))
    print(calculate_tensor_dim("tensor", 5))