import os
import numpy as np

module_dir = os.path.dirname(__file__)


def sort_teams(teams_rating, method):
    return np.array(np.argsort(teams_rating, kind=method), dtype=np.uint32)


def separate_teams(teams_id_sorted, teams_rating):
    n = teams_rating.shape[0]
    if n % 2 == 0:
        return [teams_id_sorted[i:i+2] for i in range(0, n, 2)]
    else:
        array = np.array(teams_rating[teams_id_sorted], dtype=np.uint32)
        m = int((n+1)/2)
        I = np.zeros(m, dtype=np.uint32)
        J = np.zeros(m, dtype=np.uint32)
        I[1] = array[1]-array[0]
        J[m-1] = array[n-1]-array[n-2]
        for k in range(1, m):
            I[k] = I[k-1] + (array[2*(k-1)+1] - array[2*(k-1)])
        for k in range(m-2, -1, -1):
            J[k] = J[k+1] + (array[2*(k+1)]-array[2*(k+1)-1])
        min_k = 2*np.argmin(I + J)
        teams_even = np.delete(teams_id_sorted, min_k, 0)
        return [teams_even[i:i+2] for i in range(0, n-1, 2)]


if not os.path.exists(module_dir + "/SivolobtsevRM_task_1_team_pairs"):
    os.makedirs(module_dir + "/SivolobtsevRM_task_1_team_pairs")


for root, dirs, files in os.walk(module_dir):
    if "test" in root:
        for file in files:
            if file == "players.txt":
                players = np.loadtxt(root + '/' + file, dtype=np.uint16, usecols=(1,))
            elif file == "teams.txt":
                f = open(root + '/' + file, 'r')
                col = len(f.readline().split(' '))
                f.close()
                teams_players = np.loadtxt(root + '/' + file, dtype=np.uint32, usecols=(range(1, col)))
            else:
                break

        teams_players = players[teams_players] # replace player_id->player_rating
        teams_rating = np.sum(teams_players, axis=1, dtype=np.int32) # finding team_rating lika a sum of player rating
        answer = separate_teams(sort_teams(teams_rating, "quicksort"), teams_rating)

        with open(module_dir + "/SivolobtsevRM_task_1_team_pairs/" + os.path.basename(root) + "_pairs.txt", "w") as f:
            f.write("\n".join(" ".join(map(str, x)) for x in answer))
