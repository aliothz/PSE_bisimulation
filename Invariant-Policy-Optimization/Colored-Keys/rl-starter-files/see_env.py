import utils

r = utils.make_env("MiniGrid-ColoredKeysRed-v0", 23)
g = utils.make_env("MiniGrid-ColoredKeysGreen-v0", 23)

r.reset()
g.reset()

print("red agent :", r.agent_pos)
print("green agent:", g.agent_pos)

# 如果你把 door/obstacle/key 的位置临时存成 self.xxx_pos，
# 这里一并 print
# 一样的