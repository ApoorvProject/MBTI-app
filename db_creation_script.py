import sqlite3
import os

conn = sqlite3.connect('personality_data.sqlite')
print("Opened database successfully")
c = conn.cursor()
c.execute('DROP TABLE IF EXISTS livedata_db')
c.execute('CREATE TABLE livedata_db'\
          ' (textual_input TEXT, personality_type TEXT, date TEXT)')
print("Created table successfully")

entry1 = """I'm 23 years old, Male, Unemployed, Trying to finish up school. I was described by my psychiatrist to be an "emotionally latte" person. I'm diagnosed with ADHD and depression clinically but whether those even exist or impact my mental stability is unsure. Normal upbringing, lots of friends, not religious or strict. Very supportive and loving parents Not necessarily lonely but extremely bored most likely. I enjoy sports and outdoor activities more than indoor activities atleast currently, since i don't have anything interesting to do indoors.
Nothing really comes into mind I wouldnt mind being in a leadership position. The only thing that would worry me is if i am good for this position or not. My leadership style would be looking after my workers happiness. Making sure things get done but also making sure everybodys happy doing their job and not in danger of burnout. I think i have good coordination skills. I make music and play piano as a hobby. Past is in the past, i dont really look back on it. I try to enjoy the present as much as i can I can think of my future and the things i would need to do to make it "good" but i dont act on it really.
Depends who asks and when, if i agree to help it means that i owe that person a favor, i benefit from it somehow or i just have free time to do something simple
Dont really think about it, so probably no. Don't really think about that either. If theres something i absolutely need to do then i do it as fast and as easy as possible.
No, i dont believe i do. Gym, music Gym is benefical to health and overall sense of self, music is just beautiful and i enjoy it alot.
I'm fine with any style as long as the subject interests me. Otherwise i won't learn no matter what learning style i would use.
I postpone everything until i have no time to do them Personally, just have some source of happiness in my life. Professionally, none. Maybe sometime in the future i will have something. I daydream mostly when listening to music or in stressful situations when i dissociate. I'm aware of my surroundings though. Thinking what am i doing there, why / how did i end up there and probably start figuring out how to get out of there. Usually it takes a while, but i rarely change my mind once i've made a decision. Havent really thought about it ever  If theres a situation where it benefits me to not start any pointless conflict then yes i would agree on something just to appease. Sometimes, if breaking the rule doesnt hurt other people.
"""
personality_type_entry1 = 'INTP'

c.execute("INSERT INTO livedata_db"\
          " (textual_input, personality_type, date) VALUES"\
          " (?, ?, DATETIME('now'))", (entry1, personality_type_entry1))
print("Entry 1 fed into the database")

entry2 = """ I was raised in a conservative family. Everyone just blindly followed their elders and didn’t question 
anything. I did the same till I was 10 or 11 years old. Until I realised I didn’t want to end up being a clone. I questioned
everything and only adopted things if I find them meaningful or logical. It will be interesting. I will spend my entire weekend
thinking and observing. I will learn something new. It will clear my mind. I don’t like sports. I’m terrible at it. I don’t
prefer physical activities. I love to read, daydream, learn something or just think random philosophy. Though I can be good
at sports if I’m interested in it.I’m extremely curious. I want to know things on a deep level. My ideas are usually theories
or philosophies. I don’t think you can apply them. Like patterns of the way, people behave etc.No, I don’t want to be a leader or a follower. I can be a good leader if I want to. I don’t really like being held responsible for others. If I end up being a leader, I’ll see through each person’s strengths and weaknesses and assign them with work they need to do and will give them the complete freedom to use their way, all I need is the outcome on time. Or else they will deal with the consequences. I do appreciate art. I tried drawing but lost interest as I wasn’t able to reach my desired outcome and felt discouraged. Though I intend to learn it when I get free time. I love arts that have deep meaning in them. The more you look the more amazed you are by its beauty. I don’t have much opinion about any of them. It is always best to look at the present as there is no point in looking at the other two. However, it’s hard to be in the present. I keep daydreaming all the time about useless things.I do help others as they have asked me to help them and I don’t know how to say no. I feel if I didn’t help them now, then in the future when I need help they won’t help me either and I don’t want to be helpless and lost.I do try to control the people I know like my parents or my best friend apart from them I don’t think I control people. The reason is different for each situation. I haven’t thought about this. I think it’s because I want things to be in this way or I get annoyed and tell them to do it. I love learning new things. It’s useful, helps me to develop into a better person. I like writing out ways I can be a better way; I don’t know if it can be considered as a hobby but I do this almost every day. My learning style is to connect things and to see how they are related. I like learning in-depth. I do not like loud noisy environments. I can’t hear myself think how am I supposed to study then? My struggle is I like going in-depth and it takes time so the portions in school go even faster and I have to end up learning by heart. I prefer logic because you can always remember, it’s useful in the long term. I usually make a lot of plans but I procrastinate and end up doing it just before the deadline. I do break up projects into small tasks especially if I’m stressed and don’t know what to do and where tostart. I won’t be able to handle too many things at a time I intend to reach everything I have dreamt of. To be the best version of myself and to reach my highest potential. I have a lot of potentials I can reach great heights.My fears are if I didn’t reach my dreams and didn’t get the future I wanted. It's better dying than not reaching anywhere in life. I hate public speaking and don’t like people staring at me. I don’t like spotlights. I don’t know why I hate it. Anxiety? Fear of being judged? I’m not at all attached to reality. I’m always in my head. I daydream a lot. I tried to be in the present but failed. I love daydreaming and I feel that is more important. I don’t pay attention to things around me and only do when it stands out from the rest. Due to this, there are times when I look at things as though I am seeing them for the first time because I haven’t noticed the details before. I will sit and think. What am I doing in this place? And things like that.I do have deep feelings. I think I feel things more deeply than others but I don’t like showing them. I see it as a weakness but I can’t hide the fact that I feel and care but I wish I didn’t. No, I will not agree with anyone if I don’t feel it is right. But sometimes there is no point arguing you are right, so it is better to keep your mouth shut.I don’t break rules or follow them. If in a situation rules are useful then I’ll use them. Yes,  the authority should be challenged if they are wrong. It is all about who or what is right and doing that even if it breaks the rule.
"""
personality_type_entry2 = 'INFP'

c.execute("INSERT INTO livedata_db"\
          " (textual_input, personality_type, date) VALUES"\
          " (?, ?, DATETIME('now'))", (entry2, personality_type_entry2))
print("Entry 2 fed into the database")

conn.commit()
conn.close()