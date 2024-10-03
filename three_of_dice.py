dice1, dice2, dice3 = map(int, input().split(' '))
dice = [dice1, dice2, dice3]
set_dice =set([dice1, dice2, dice3])


if len(set_dice) ==1:
    print(f'{10000 +dice1*1000}')
elif len(set_dice) ==2:
    if dice1 == dice2 or dice1 == dice3: 
        print(f'{1000 +dice1*100}')
    else: print(f'{1000 +dice2*100}')
else:
    print(max(dice1, dice2, dice3)* 100)