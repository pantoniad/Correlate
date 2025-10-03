import test_class

x = 10
y = 1
z = 2

# Now printing x
test_x = test_class.Test_class()
test_x.print_x(x)

# Now printing y  
test_y = test_class.Test_class.Test_class_inner(y)
test_y.print_y()

# Now printing z  
test_z = test_class.Test_class.Test_class_inner.Test_class_inner_2(z)
test_z.print_z()
 