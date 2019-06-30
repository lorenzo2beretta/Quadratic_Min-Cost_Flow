# Quadratic_Min-Cost_Flow
This report is devoted to the project of "Computational Mathematics for 
learing and data analysis", prof. Frangioni and Poloni, UniPi 2018/19.

This is a solver for quadratic separable min-cost flow problem. It exploits
a dual approach through KKT conditions reducing the problem to a linear
system of the form Sx = b where S is a symmetric structured matrix.
Then it employs the conjugate gradient method exploiting the strucutre
of S to efficiently solve it.

This work is enriched with some experminetal benchmark assessing its
feasibiity in practice.