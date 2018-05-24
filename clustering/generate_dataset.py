"""!

@brief Filip Hörnsten's Bachelor degree project 2018

@authors Filip Hörnsten
@date 2018
@copyright GNU Public License
"""

import random

def generate_dataset():
        # Maintain a list of already generated numbers to make sure they are disjoint from each other
        generatedNumbers = []

        # Set the seed for reproducibility 
        random.seed(92385736)
        #random.seed(6139866599598374)
        # Bad: 3.77686 4.342717
        # Good: 1.602743 3.620128
        # Position: 960
        # Open the file, create one if it doesn't exist already
        file= open("scenario5.data","w+")
        
        for i in range(1000):
                # Randomly generated points
                x = round(random.uniform(0.5, 7.5), 6)
                y = round(random.uniform(0.5, 7.5), 6)

                #print(generatedNumbers)
                # Check that the generated point is not already in the list of generated numbers
                while([x,y] in generatedNumbers):
                        print("Duplicate found:", x, y)
                        x = round(random.uniform(0.5, 7.5), 6)
                        y = round(random.uniform(0.5, 7.5), 6)
                
                generatedNumbers.append([x, y])
                
                # Format the output string
                point = str(x) + " " + str(y) + "\n"
                #print(point)
                
                # Write to the file
                file.write(point)

        # Close the file
        file.close()

generate_dataset()           
# EOF
