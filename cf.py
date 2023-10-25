from time import time
import psutil


class Employee():
    def __init__(self, personal_number, sup, lang):
        self.personal_number = personal_number
        self.sup = sup
        self.lang = lang

        self.bariers = {}

        if sup is None:
            self.bariers = {"A": 0, "B": 0}
        elif lang == "A":
            self.bariers[lang] = 0
            self.bariers["B"] = sup.bariers["B"] + 1
        else:
            self.bariers[lang] = 0
            self.bariers["A"] = sup.bariers["A"] + 1

def solve():
    n = int(input())
    langs = [i for i in input().split()]
    people = [int(i) for i in input().split()]
    process = psutil.Process()
    print(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024} MB")

    start = time()
    n = 10**6
    langs = ["A"]*n
    people = [i for i in range(n+1)] + [i for i in range(n, -1, -1)]
    employees = {}

    sup = Employee(0, None, "AB")
    for i in range(1, len(people)):
        personal_number = people[i]
        lang = langs[personal_number - 1]
        if personal_number == 0:
            continue
        elif personal_number == sup.personal_number:
            sup = sup.sup
        else:
            employee  = Employee(personal_number, sup, lang)
            employees[personal_number] = employee
            sup = employee

    bariers = []
    for i in range(1, n + 1):
        employee = employees[i]
        bariers.append(str(employee.sup.bariers[employee.lang]))

    #print(" ".join(bariers))
    print(time()-start)
    print(f"Final memory usage: {process.memory_info().rss / 1024 / 1024} MB")

solve()
