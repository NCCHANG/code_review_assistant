import os
import pandas as pd
from git import Repo

def mine_local_repository(repo_url, repo_dir):
    print(f"\n📁 Processing repository: {repo_url}")
    
    # 1. Clone the repo to your machine if it isn't there already
    if not os.path.exists(repo_dir):
        print(f"   Downloading (cloning) the repository... This takes a minute or two.")
        Repo.clone_from(repo_url, repo_dir)
    
    repo = Repo(repo_dir)
    print(f"   Successfully loaded. Mining commit history at lightning speed...")
    
    dataset = []
    keywords = ['fix', 'bug', 'resolve', 'patch', 'issue']
    
    # 2. Iterate through the last 10,000 commits! (No API limits here)
    for commit in list(repo.iter_commits('HEAD', max_count=10000)):
        message = commit.message.lower()
        
        # 3. Check if the commit message implies a bug fix
        if any(keyword in message for keyword in keywords):
            if not commit.parents:
                continue
                
            parent = commit.parents[0]
            
            # 4. Find what files changed between the buggy parent and this fix
            diffs = parent.diff(commit)
            
            for diff in diffs:
                # We only want Python files that were modified
                if diff.a_path and diff.a_path.endswith('.py') and diff.change_type == 'M':
                    try:
                        # Extract buggy and fixed code directly from your hard drive!
                        buggy_code = parent.tree[diff.a_path].data_stream.read().decode('utf-8')
                        fixed_code = commit.tree[diff.b_path].data_stream.read().decode('utf-8')
                        
                        if buggy_code and fixed_code and buggy_code != fixed_code:
                            dataset.append({
                                'input_text': buggy_code,
                                'target_text': fixed_code,
                                'commit_message': commit.message.split('\n')[0],
                                'repo': repo_url.split('/')[-1]
                            })
                    except Exception:
                        # Skip files that can't be decoded cleanly
                        continue

    print(f"   ✅ Extracted {len(dataset)} Python bugs from this repository!")
    return dataset

if __name__ == "__main__":
    # We are targeting three massive, highly respected Python projects
    target_repos = [
        ("https://github.com/psf/requests.git", "./cloned_repos/requests"),
        ("https://github.com/pallets/flask.git", "./cloned_repos/flask"),
        ("https://github.com/django/django.git", "./cloned_repos/django")
    ]
    
    all_bugs = []
    
    # Ensure our download directory exists
    os.makedirs("./cloned_repos", exist_ok=True)
    
    for url, path in target_repos:
        bugs = mine_local_repository(url, path)
        all_bugs.extend(bugs)
        
    # Save the massive dataset!
    if all_bugs:
        df = pd.DataFrame(all_bugs)
        output_name = "massive_local_bugs_dataset.csv"
        df.to_csv(output_name, index=False)
        print(f"\n🎉 SUCCESS! Saved {len(df)} total bugs to {output_name}")
        print("Zero API rate limits hit! 😎")