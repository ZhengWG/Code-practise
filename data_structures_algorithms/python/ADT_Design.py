#codeing=utf-8
"""
ADT_Desing:抽象数据类型设计
说明：需要记录学校中的两类人员：
    + 学生和教职工，两者具备一些共性，如姓名，性别，年龄等
    + 学生和教职工有各自的专属信息
设计：需要设计基本人员ADT，作为学生和教职工的父类
"""

import datetime


# 派生专用的异常类
class PersonTypeError(TypeError):
    pass


class PersonValueError(ValueError):
    pass


class Person:
    """
    ADT Person:
    Person(self, strname, strsex, tuple birthday, str ident)
    id(self)
    name(self)
    sex(self)
    birthday(self)
    age(self)
    set_name(self, str name)
    <(self, Person another)
    details(self)
    """
    _num = 0

    def __init__(self, name, sex, birthday, ident):
        if not (isinstance(name, str) and sex in ("女", "男")):
            raise PersonValueError(name, sex)
        try:
            # 生成一个 日期对象
            birth = datetime.date(*birthday)
        except Exception:
            raise PersonValueError("Wrong date:", birthday)
        self._name = name
        self._sex = sex
        self._birthdary = birth
        self._id = ident
        # 添加一个Person实例
        Person._num += 1

    def id(self): return self._id

    def name(self): return self._name

    def sex(self): return self.sex

    def birthday(self): return self._birthdary

    def age(self): return (datetime.date.today().year - self._birthdary.year)

    def set_name(self, name):
        # 修改名字
        if not isinstance(name, str):
            raise PersonValueError("set_name:", name)
        self.name = name

    def __lt__(self, another):
        # ’<‘函数
        if not isinstance(another, Person):
            raise PersonTypeError(another)
        return self._id < another._id

    @classmethod
    def num(cls): return Person._num

    def __str__(self):
        return " ".join((self._id, self._name, self._sex, str(self._birthdary)))

    def details(self):
        return " ".join(("编号：" + self._id, \
                            "姓名：" + self._name, \
                                "性别" + self._sex, \
                                    "出生日期：" + str(self._birthdary)))


# p1 = Person("张三", "女", (1995, 7, 30), "12312321")
# p2 = Person("李四", "男", (1993, 7, 30), "00000000")
# p_list = [p1, p2]
# print(p1)
# print(p1.details())
# print(p_list)
# p_list.sort()
# print(p_list)

class Student(Person):
    """
    Student(self, strname, strsex, tuple birthday, str department)
    department(self)
    en_year(self)
    scores(self)
    set_course(self, str course_name)
    set_score(self, str course_name, int score)
    """
    _id_num = 0

    @classmethod
    def _id_gen(cls):
        # 实现学生id的生成
        cls._id_num += 1
        year = datetime.date.today().year
        return "1{:04}{:05}".format(year, cls._id_num)

    def __init__(self, name, sex, birthday, department):
        super().__init__(name, sex, birthday, Student._id_gen())

        self._department = department
        self._enroll_date = datetime.date.today()
        self._courses = {}

    def set_course(self, course_name):
        self._courses[course_name] = None

    def set_score(self, course_name, score):
        if course_name not in self._courses:
            raise PersonValueError("No this course selected:", course_name)
        self._courses[course_name] = score

    def scores(self):
        return [(cname, self._courses[cname]) for cname in self._courses]

    def details(self):
        # 重写details函数
        return " ".join((super().details(), \
            "入学日期：" + str(self._enroll_date), \
                "院系：" + self._department, \
                    "课程记录:" + str(self.scores())))


# p1 = Student("张三", "女", (1995, 7, 30), "数学"")
# p1.set_course('数学')
# p1.set_score('数学', 100)
# print(p1.scores())
# print(p1.details())


class Staff(Person):
    """
    Staff(self, strname, strsex, tuple birthday, tuple entry_date)
    department(self)
    salary(slef)
    entry_date(self)
    position(self)
    set_salary(self, int amount)
    """
    _id_num = 0

    @classmethod
    def _id_gen(cls, birthday):
        # 实现职工号生成
        cls._id_num += 1
        birth_year = datetime.date(*birthday).year
        return "0{:04}{:05}".format(birth_year, cls._id_num)

    def __init__(self, name, sex, birthday, entry_date=None):
        super().__init__(name, sex, birthday, Staff._id_gen(birthday))
        if entry_date:
            try:
                self._entry_date = datetime.date(*entry_date)
            except Exception:
                raise PersonValueError("Wrong date:", entry_date)
        else:
            self._entry_date = datetime.date.today()
        # 基本工资
        self._salary = 1720
        self._department = "未定"
        self._position = "未定"

    def set_salary(self, amount):
        if not type(amount) is int:
            raise PersonTypeError("Wrong type for salary:", amount)
        self._salary = amount

    def set_position(self, position):
        self._position = position

    def set_department(self, department):
        self._department = department

    def details(self):
        return " ".join((super().details(), \
            "入职日期：" + str(self._entry_date), \
                "院系：" + self._department, \
                    "职位：" + self._position, \
                        "工资：" + str(self._salary)))


p1 = Staff("张三", "女", (1995, 7, 30))
p1.set_department('数学')
p1.set_position("教授")
p1.set_salary(20000)
print(p1.details())