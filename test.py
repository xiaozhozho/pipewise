import pandas as pd 
import pipewise 



if __name__ == "__main__":
    df = pd.DataFrame({
        'ac': [1, 2, 3, 4],
        'bc': [5, 6, 7, 8],
        'lc': [[1,2,3], [2,5,1], [7,4,2], [20,34,12]],
        'sc': ['a,b,c', 'b,2,e', 'c,3,5', 'd,2,3']
        })
    pipe = pipewise.Pipewise(df)
    
    @pipe.register(outputs=['lc_first','sc_list'])
    def proc_string_as_list(lc,sc):
        return lc[0], sc.split(',')
    
    df_result = pipe.run()
    print(df_result)