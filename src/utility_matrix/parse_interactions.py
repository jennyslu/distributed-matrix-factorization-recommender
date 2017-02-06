import json
import os
import sys
from multiprocessing import Pool

def parse_utility_matrix(filenames):
    global user_map
    global project_map
    for filename in filenames:
        in_file_path = 'new_input/' + filename
        # output file
        out_file = 'new_output/' + filename + '.txt'
        writer = open(out_file, 'w')
        # parse file
        with open(in_file_path) as in_file:
            next(in_file)
            for line in in_file:
                #decode
                [event, project, user] = line.rstrip().split(',')
                i = user_map[user]
                j = project_map[project]
                writer.write('{},{}\n'.format(i,j))
        writer.close()

def create_project_mappings():
    # read in all projects
    projects = []
    with open('input/projects.csv') as f:
        for line in f:
            projects.append(line.split(',')[0])
    # create dictionary for mapping
    project_map = {k: v for (k, v) in zip(projects[1:], range(1,len(projects)))}
    return project_map

def create_user_mappings():
    # read in all users
    users = []
    user_files = ['input/users.csv', 'input/users1.csv']
    for user_file in user_files:
        with open(user_file) as f:
            next(f)
            for line in f:
                users.append(line.rstrip())
    # create dictionary for mapping
    user_map = {k: v for (k, v) in zip(users, range(1,len(users)+1))}
    return user_map

def originators(project_map, user_map):
    writer = open('output/originals.txt', 'w')
    last_user_id = len(user_map)
    with open('input/projects.csv') as f:
        next(f)
        for line in f:
            repo_name = line.rstrip().split(',')[0]
            user, repo = repo_name.split('/')
            j = project_map[repo_name]
            # if exists in users list
            try:
                i = user_map[user]
            # else add to end
            except KeyError:
                last_user_id += 1
                user_map[user] = last_user_id
                i = last_user_id
            # write to text file
            writer.write('{},{}\n'.format(i,j))
    writer.close()
    return user_map

if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('iso8859-1')

    # read in JSON file of all pull requests for given year for projects
    filenames = os.listdir('new_input')

    project_map = create_project_mappings()
    user_map = create_user_mappings()

    # create a file that outputs pairs of creators and their repos
    # returns updated user_map
    user_map = originators(project_map, user_map)

    p = Pool(8)
    p.map(parse_utility_matrix, (filenames, ))
