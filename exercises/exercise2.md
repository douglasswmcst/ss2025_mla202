## Case Study: Mobile Legends.

Image you are an agent -> Layla. 

Agent State -> 80% HP, 100% Mana, Spell 1 & 2 available

Full Environment -> Whole Map

Partial Observable -> Mini-map

Allies Position
Enemy not in fog
Turtle status

Agent Observable Scope -> Mid lane

Global Environment State -> 2 Mins: 45 sec has passed.

Time: 2 Mins:45 sec has passed.

Objective: Tier 1 Tower Full HP

Opposing Agent: Miya 50% HP

Creeps: 4 melee, 1 range available

Possible Actions:

1. Take objective and creeps

2. Be aggresive and help turtle (possible to kill jungler)

3. farm in lane and play safe. (ignore, and take tower gold)

Rewards
1. Turtle : Team Gold +50gold
2. Kill creeps : +20 gold
3. kill miya +200 gold
4. chip tower gold hp +45 gold

1. Policies & Control
- Policy: Farm in lane and play safe. (ignore, and take tower gold)

2. Eagerness via discount factor
   1. Discount factor: 0.9
   2. Eagerness: Agent is eager to take short term gains over long term gains.
      1. Help to take turtle and possible kill jungler (long term gain)
      2. Farm in lane and take tower gold (short term gain)
   3. Discount factor: 0.1
   4. Eagerness: Agent is eager to take long term gains over short.
   Try to kill miya when miya trys to go to turtle.









