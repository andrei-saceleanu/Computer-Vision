# Homework 3 Computer Vision
#### Saceleanu Andrei-Iulian, IA1-B

### Line detection

- results/lines

- implemented using Hough Line transform in polar form

- only lines that have votes above a certain threshold are kept and plotted

- the line is available as the segment from x=0 to x=width of image

### Rectangle detection

- results/rectangles

- from the detected lines, pairs of parallel lines are obtained

- next, the pairs of parallel lines are paired with an orthogonal pair
to form a rectangle(approximations are allowed, such as 89 degrees instead of 90)

- the rectangles are available through a clockwise ordering of its corners (i1, i2, i3, i4)
