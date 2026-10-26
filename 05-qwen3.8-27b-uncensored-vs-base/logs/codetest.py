def group_anagrams(words):
    groups = {}
    for word in words:
        key = ''.join(sorted(word))
        groups.setdefault(key, []).append(word)
    return list(groups.values())


r=group_anagrams(['eat','tea','tan','ate','nat','bat']); s=sorted(sorted(g) for g in r); assert s==sorted([sorted(x) for x in [['ate','eat','tea'],['nat','tan'],['bat']]])
print('OK_TEST_PASSED')
