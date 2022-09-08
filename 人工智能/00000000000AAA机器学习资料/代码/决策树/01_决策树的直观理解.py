# -- encoding:utf-8 --
"""
Create on 19/3/10
"""


def predict(x1, x2, x3):
    if x3 >= 97.5:
        return 1
    else:
        if x1 == 1:
            return 1
        else:
            if x2 == 0:
                return 0
            elif x2 == 1:
                return 1
            else:
                return 0

# 无房产、单身、年收入55K
predict(0,0,55)