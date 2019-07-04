# Quadratic_Min-Cost_Flow

This report is devoted to the project of "Computational Mathematics for 
learing and data analysis", prof. Frangioni and Poloni, UniPi 2018/19.

This is an implementation of Conjugate Gradient method employed to solve
quadratic separable min-cost flow problem. It exploitsa dual approach
through KKT conditions reducing the problem to a linear system of the
form Sx = b where S is a symmetric structured matrix.

How to use this code?

You can find all the relevant methods in cg.py, they are well documented.
Moreover you can find some utility function in test.py, basically they are
function I employed to automate experimentation tasks.

You can find attached, within folder graph, some graphs in DIMACS format
employed in experimentations. Moreover you can find data1.csv and data2.csv
containing experiment's results.