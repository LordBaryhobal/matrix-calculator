#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
from typing import Union

import numpy as np


TOLERANCE = 1e-15

class EquationParser:
    def __init__(self) -> None:
        self.matrix = Matrix(0, 0)
        self.constants = Vector(0)
        self.nEquations = 0
        self.nVariables = 0

    def parse(self, equations: list[str]) -> None:
        self.nEquations = len(equations)
        varNames = set()
        coeffs = []
        constantsArray = []

        # https://regex101.com/r/kBY2E9/2
        regex = re.compile(r"(?P<coef>([+-]\s*)?(\d+(\.\d+)?(\s*/\s*\d+(\.\d+)?)?)?)\s*\*?\s*(?P<var>[a-zA-Z]+\d*)")
        for i, equ in enumerate(equations):
            variables, constant = equ.split("=")
            constantsArray.append(self.computeCoef(constant))
            matches = regex.finditer(variables)
            coeffsDict = {}

            for m in matches:
                coef = re.sub(r"\s+", "", m.group("coef"))
                coef = self.computeCoef(coef)

                var = m.group("var")
                varNames.add(var)

                coeffsDict[var] = coef

            coeffs.append(coeffsDict)

        self.nVariables = len(varNames)
        matrixArray = np.zeros((self.nEquations, self.nVariables))
        varNames = sorted(list(varNames))

        for i, var in enumerate(varNames):
            for j in range(self.nEquations):
                coef = coeffs[j][var]
                matrixArray[j, i] = coef

        self.matrix = Matrix.fromArray(matrixArray)
        self.constants = Vector.fromArray(constantsArray)

    def computeCoef(self, coefStr: str) -> float:
        coefStr = coefStr.strip()
        if coefStr in ("+", ""):
            return 1

        if coefStr == "-":
            return -1

        if "/" in coefStr:
            a, b = coefStr.split("/")
            return float(a) / float(b)

        return float(coefStr)

class Vector:
    def __init__(self, size: int) -> None:
        self.size = size
        self.values = np.zeros([size])

    def fromArray(array: Union[list, np.array]) -> "Vector":
        a = np.array(array)
        size = a.shape[0]
        v = Vector(size)
        v.values = a
        return v

    def __str__(self):
        return "(" + ",".join([f"{v + 0:.6g}" for v in self.values]) + ")"

    def __repr__(self):
        return "Vector" + str(self)


class Matrix:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.mat = np.zeros([height, width])
    
    def fromArray(array: Union[list, np.array]) -> "Matrix":
        mat = np.array(array)
        width = mat.shape[1]
        height = mat.shape[0]
        m = Matrix(width, height)
        m.mat = mat
        return m
    
    def swapRows(self, i: int, j: int) -> None:
        self.mat[i], self.mat[j] = self.mat[j], self.mat[i]
    
    def addRow(self, i: int, j: int, f: float = 1) -> None:
        self.mat[i] += self.mat[j] * f
        self.mat[np.abs(self.mat) < TOLERANCE] = 0
    
    def multiply(self, i: int, f: float) -> None:
        self.mat[i] *= f
        self.mat[np.abs(self.mat) < TOLERANCE] = 0
    
    def divide(self, i: int, f: float) -> None:
        self.mat[i] /= f
        self.mat[np.abs(self.mat) < TOLERANCE] = 0

    def prettyPrint(self, extended: bool = False) -> None:
        width = self.width * 11
        if extended:
            width += 1

        print("┌" + " " * width + "┐")

        for y in range(self.height):
            print("│", end="")
            w = self.width
            if extended:
                w -= 1

            for x in range(w):
                print(f"{self.mat[y, x] + 0: 10.6g} ", end="")

            if extended:
                print(f"│ {self.mat[y,-1] + 0: 10.6g}", end="")

            print("│")

        print("└" + " " * width + "┘")


class SolutionType:
    INDETERMINATE = -1
    IMPOSSIBLE = 0
    UNIQUE = 1
    INFINITE = 2
    
    def name(value: int) -> str:
        for k, v in SolutionType.__dict__.items():
            if v == value:
                return k
        
        return "<unknown>"


class Gaussificator:
    def __init__(self, rawMatrix: Matrix, constants: Vector) -> None:
        self.rawMatrix = rawMatrix
        self.constants = constants
        
        if self.rawMatrix.height != self.constants.size:
            raise ValueError("The input matrix and constants vector do not have the same height")
        
        a = np.zeros([self.rawMatrix.height, self.rawMatrix.width+1])
        a[:, :-1] = self.rawMatrix.mat
        a[:, -1] = self.constants.values
        
        self.combMat = Matrix.fromArray(a)
        self.nEquations = self.rawMatrix.height
        self.nVariables = self.rawMatrix.width
    
    def findNextNonNull(self, x: int = 0, startY: int = 0) -> int | None:
        for y in range(startY, self.nEquations):
            if self.combMat.mat[y, x] != 0:
                return y
        
        return None
    
    def echelon(self) -> Matrix:
        iterations = min(self.nEquations, self.nVariables)
        # Gauss
        for i in range(iterations):
            pivot = self.combMat.mat[i, i]
            
            # Make pivot 1
            if pivot != 0:
                self.combMat.divide(i, pivot)
            
            else:
                y = self.findNextNonNull(i, i)
                if y is not None:
                    self.combMat.addRow(i, y, 1 / self.combMat.mat[y, i])
            
            # Make all rows below 0
            for y in range(i+1, self.nEquations):
                self.combMat.addRow(y, i, -self.combMat.mat[y, i])
        
        # Jordan
        for y in range(iterations-2, -1, -1):
            for x in range(y+1, iterations):
                self.combMat.addRow(y, x, -self.combMat.mat[y, x])
        
        return self.combMat


class Solution:
    def __init__(self) -> None:
        self.type: int = SolutionType.INDETERMINATE
        self.values: Union[list[Vector], None] = None
        self.matrix: Union[Matrix, None] = None

    def explain(self) -> None:
        if self.type == SolutionType.INDETERMINATE:
            print("The solution is indeterminate")
        elif self.type == SolutionType.IMPOSSIBLE:
            print("There are no solutions")
        elif self.type == SolutionType.UNIQUE:
            print("There is a unique solution")
            self.printSolutions()
            self.printSolutions(True)
        else:
            print("There are infinite solutions")
            self.printSolutions()
            self.printSolutions(True)

    def printSolutions(self, separated=False) -> None:
        nVar = self.values[0].size

        if not separated:
            vars = [f"x{i+1}" for i in range(nVar)]
            print(f"({','.join(vars)}) = ", end="")

            print(self.values[0], end="")

            for i, vec in enumerate(self.values[1:]):
                param = chr(ord("a") + i)
                print(f" + {param} * {vec}", end="")

            print()

        else:
            for i in range(nVar):
                print(f"x{i+1} = ", end="")

                allZero = tuple(v.values[i] for v in self.values) == (0,) * len(self.values)

                if allZero:
                    print(chr(ord("a") + i))
                    continue

                else:
                    s = ""
                    for j, vec in enumerate(self.values):
                        param = chr(ord("a") + j - 1) if j != 0 else ""
                        val = vec.values[i]
                        if val == 0:
                            continue

                        sign = "-" if val < 0 else "+"
                        if s != "":
                            s += f" {sign} "

                        if j == 0 or val != 1:
                            s += f"{abs(val):.6g}"

                        s += param

                    if s == "":
                        s = 0

                    print(s)



class Solver:
    def __init__(self, rawMatrix: Matrix, constants: Vector) -> None:
        self.rawMatrix = rawMatrix
        self.constants = constants
        self.solution = Solution()
    
    def solve(self) -> Solution:
        sol = self.solution
        gauss = Gaussificator(self.rawMatrix, self.constants)
        matrix = gauss.echelon()
        sol.matrix = matrix
        
        nEqu = gauss.nEquations
        nVar = gauss.nVariables
        
        sol.type = SolutionType.UNIQUE
        
        for y in range(nEqu):
            allZero = True
            
            for x in range(nVar):
                if matrix.mat[y, x] != 0:
                    allZero = False
                    break
            
            if allZero:
                if matrix.mat[y, nVar] == 0:
                    if y < nEqu:
                        sol.type = SolutionType.INFINITE
                else:
                    sol.type = SolutionType.IMPOSSIBLE
                    break
        
        if sol.type == SolutionType.UNIQUE and nEqu < nVar:
            sol.type = SolutionType.INFINITE
        
        if sol.type == SolutionType.UNIQUE:
            sol.values = Vector.fromArray(matrix.mat[:, -1])
        
        elif sol.type == SolutionType.INFINITE:
            vectors = []
            constants = Vector(nVar)
            constants.values[:nEqu] = matrix.mat[:, -1]
            vectors.append(constants)

            for i in range(nVar):
                # If param
                if i >= nEqu or matrix.mat[i, i] == 0:
                    vec = Vector(nVar)
                    vec.values[:nEqu] = -matrix.mat[:, i]
                    vec.values[i] = 1
                    vectors.append(vec)

            sol.values = vectors

        return sol


if __name__ == "__main__":
    parser = EquationParser()
    parser.parse([
        "3x1 - 3x2 - x3 + 2x4 - 9x5 = 13",
        "x1 - x2 + 2x3 - x4 - 6x5 = -6",
        "x1 - x2 + x3 + x4 - 6x5 = 1",
        "-x1 + x2 - x3 - 2x4 + 7x5 = -3"
    ])

    """
    A = Matrix.fromArray([
        [3, -3, -1, 2, -9],
        [1, -1, 2, -1, -6],
        [1, -1, 1, 1, -6],
        [-1, 1, -1, -2, 7]
    ])
    
    b = Vector.fromArray([
        13,
        -6,
        1,
        -3
    ])
    """

    solver = Solver(parser.matrix, parser.constants)
    #solver = Solver(A, b)
    solution = solver.solve()

    solution.matrix.prettyPrint(True)
    solution.explain()

    #(x1,x2,x3,x4,x5) = (2,0,-3,2,0) + a * (1,1,0,0,0) + b * (3,0,2,1,1)