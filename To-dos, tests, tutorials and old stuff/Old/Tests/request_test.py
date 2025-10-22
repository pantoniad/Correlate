from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()  # or Edge, Firefox, etc.
driver.get("https://aes.propulsion.gr/2spoolboost.html")

time.sleep(5)  # Wait for JS to load everything

# Example: find a button or input by ID, name, or class
# Adjust selectors based on actual HTML
input_elem = driver.find_element(By.ID, "someInputField")
input_elem.send_keys("42")

# Get page title
print(driver.title)

driver.quit()