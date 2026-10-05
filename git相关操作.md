//git相关操作的代码//

查看文件夹中各文件的状态（是否进入暂存区/提交历史区等）：git status
把文件提交到暂存区：git add
把文件保存到提交历史区：git commit -m “备注”
查看提交历史区有哪些已经保存了的文件：git log（git log --oneline：只占一行，不显示作者等复杂信息）

//回退版本的3+1种办法：
1.改动只发生在工作区：git restore 文件1 文件2（若要一次性把当前目录所有文件都回滚：直接用git restore .）
2.改动发生在暂存区：git restore --staged 文件1 文件2，然后再重复上一条步骤
3.改动发生在提交历史区（已经commit了）：
                                    想要直接删掉这一版的改动，即工作区和暂存区里也不留下(别的在工作区等还没提交的文件也会一并被清理掉，所以要谨慎)：git reset --hard HEAD~1
                                    想保留工作区：--mixed(默认)
                                    想保留工作区、暂存区：--soft      
4.增加一个版本，是和改动之前一个版本相同的：git revert HEAD         

//创建分支和移动分支：
1.创建分支：git branch 分支名（不加分支名，即只是查看有哪些分支、现在在的是哪个分支）
2.切换分支：git switch 分支名（切换后，可以用git log查看）
3.一步到位：git switch -c 分支名

//合并开发好的分支：
1.合并：先用git switch回到main分支，然后再git merge 分支名；
2.删除支线开发的分支：git branch -d 分支名

//开发过程，要及时把自己所做的这条分支挪到主线main最新的commit后面，不然冲突多了的话，在合并的时候不好解决：
1.git rebase main：若有冲突，先解决冲突（用vscode打开文件修改等等）
2.修改完后，用git add 文件名
3.git rebase --continue（不用git commit）

//在新分支上操作时，突然有用户给main上的文件提建议，得去改：
(法1)
1.为了保存在现在已经在改的这条分支，用git stash
2.切回main分支去改动：git switch master，然后从main分支上再拉一条分支来修bug：用git switch -c 分支名（修bug一般用fix/···），修完后正常用git add,git commit，合并好后把这条修bug的分支给删掉：git branch -d fix···（分支名）
3.回到之前正在开发中的那条分支：git stash pop（用git diff确认改动有哪些）
（法2）使用新的worktree【两个worktree公用一个提交历史commit,但是工作区和暂存区是独立的，所以最后可以回到原worktree以后，删掉改bug的worktree】
1.先新建一个worktree：git worktree add -b 分支名 ../新文件夹名（如：机器人学习组_新） main（那里的分支名就是指的新目录中的，除了main以外的用来修bug等的分支，同上）
2.在新目录（worktree）里面改动，并git add,git commit,合并进main分支（git switch main,git merge 分支名），然后删除这个改动的分支：git branch -d 分支名
3.改完之后，回到原来的worktree（机器人学习组）：cd ../机器人学习组，再删掉原来的worktree:git worktree remove ../机器人学习组_新 （不能直接拖进废纸篓，不然git那边的记录还没改）
4.回到原目录/worktree继续开发新功能，并提交：git add,git commit,最后合并到main分支：git switch main,git merge 开发新功能的分支名,最后删掉这个新功能分支：git branch -d 新功能分支名