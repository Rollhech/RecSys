# RecSys
## Course Recommendation System

A students' (elective) course selection can be understood as a sequential decision making process, where every decision influences the considerations made for the next course potentially taken. 
As early as within the first year of their academic life, students are required to take certain introductory courses that influence the course options available to them at a later stage in their studies and consequently the majors they can attain.
Often so, bachelor students are overwhelmed by the choices made available to them, such that they rely on word-of-mouth recommendations of their fellow students, rather than seizing the opportunity to explore their interests. <br>
Schwartz and Ward (2004) first introduced this phenomenon as the **Paradox of Choice** and argued that the cost of having an overabundance of choice coupled with the aim of making the best choice possible leads to reduced overall well-being of the person exposed to this decision-making scenario.<br>
<br>
To address the aforementioned implications of the course selection process and possibly thereby enhance the overall course selection experience, a Course Recommendation System can be implemented to assist students in the digital environment they interact with.
This will help students to find relevant courses to fulfil their mandatory workload, but also explore new courses they might have not been aware of otherwise. <br>
Two different approaches are explored in the scope of this project. First, **User-based Collaborative Filtering** is applied to determine similarities amongst students and hence their respective course selection.
The student data is divided into distinct groups using the **k-Means Clustering** Algorithm. The cluster to which new students potentially belong is predicted, and different measures are explored to choose the best set of course recommendations.<br>x
Collaborative Filtering based Recommendation Systems do suffer from a major disadvantage, as they enforce a so-called **Echo Chamber**. Once a similarity between users or items are determined, only items that are associated - or *similar* - are recommended. This implies that no new content is explored, hence a users’ opinion or taste is reinforced. <br>
This is the reason why the second approach deploys **Reinforcement Learning**, with which the problem is modelled with the aid of **Multi-Armed Bandits**, where the exploration-exploitation trade-off delivers the right dosage of recommending new content **(exploration)** in combination with suitable content **(exploitation)**. <br>As a consequence, diversity in the recommended items is provided, as the underlying mechanism breaks the Echo Chamber and thus operates beyond merely the association of similar items. 
All approaches are evaluated offline by comparing how accurate the Recommendation System can predict the actual course choice that was taken by respective test subjects.
