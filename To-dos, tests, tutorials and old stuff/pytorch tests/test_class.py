class Test_class():
    
    def __init__(self):
        pass
    
    @staticmethod
    def print_x(x):
        print(x)

    class Test_class_inner():

        def __init__(self, y):
            self.y = y

        def print_y(self):
            print(self.y)

        class Test_class_inner_2():

            def __init__(self, z):
                self.z = z

            def print_z(self):
                return print(self.z)

#test = Test_class.Test_class_inner.Test_class_inner_2(z = 10)
#test.print_z()