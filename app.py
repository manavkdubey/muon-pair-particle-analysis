import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit import Minuit, minimize
from scipy import integrate
from scipy import stats
import pandas as pd
from scipy.optimize import curve_fit
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource,Title



# Define the layout of the app
st.title("Muon Analysis App")
overview=['Overview','Plot']
first_options=st.sidebar.radio("",overview)
if(first_options=="Overview"):
    Data_input_list=[
                        "Invariant mass of particle pairs",
                        "Transverse momentum of muon pairs",
                        "Pseudorapidity of muons",
                        "Total momentum of muon pair",
                        "Transverse momentum of the first muon",
                        "Transverse momentum of the second muon"]
    st.write("The Particle Physics Data Analysis App is a powerful tool designed to perform comprehensive analysis on particle collision data, particularly focusing on muon properties. This app utilizes the Streamlit framework to create an interactive and user-friendly interface for physicists, researchers, and enthusiasts to explore various aspects of particle interactions.")
    st.write("**1.Data Entry**")
    st.write(Data_input_list)
    st.write("**2.Data Analysis**")
    Data_analysis_list=["The app instantly generates insightful visualizations and analysis based on the input values.",
                    "It calculates and displays key metrics, such as mean, median, and standard deviation, for each input parameter.",
                    "Interactive histograms and scatter plots provide an immediate overview of the data distribution."]
    st.write(Data_analysis_list)
    st.write("**3.Advanced Analysis**")
    st.write("Users can dive deeper into specific analysis areas, such as:")
    Advanced_analysis_list=[
                            "Kinematic distributions of muon properties",
                            "Correlations between different parameters",
                            "Trends in particle behavior at different energy levels"]
    st.write(Advanced_analysis_list)

    st.write("**4.Machine Learning Model**")
    Machine_learning_list=["The app includes a pre-trained machine learning model.","Users can upload their data for prediction and classification tasks using the ML model.","This provides an opportunity to apply predictive analytics to particle physics data."]
    st.write(Machine_learning_list)

   

    st.write("**Benefits:**")
    st.write("- **Ease of Use:** The intuitive interface allows users with varying levels of expertise to perform complex data analysis without programming knowledge.")
    st.write("- **Immediate Insights:** Users can quickly visualize and understand particle properties, making it an ideal tool for preliminary analysis.")
    st.write("- **Interactivity:** The app offers interactive features, allowing users to explore data from different angles and perspectives.")
    st.write("- **Educational Tool:** It serves as an educational resource to help students and researchers grasp particle physics concepts through hands-on exploration.")
    st.write("- **Research Support:** Researchers can use the app to validate theoretical predictions and conduct preliminary studies for more in-depth research.")

    st.write("**Usage:**")
    st.write("The Particle Physics Data Analysis App empowers researchers and students to gain insights into particle properties, relationships, and behavior with just a few clicks. Whether for educational purposes or actual research, the app provides a streamlined approach to understanding the intricacies of particle physics.Please note that this is a conceptual description of the app's features and benefits. Developing the actual app with these functionalities would involve coding and implementing these features using Streamlit and possibly integrating machine learning libraries for model deployment.")
if(first_options=="Plot"):


    # File upload section
    st.sidebar.title("Data Import")
    file = st.sidebar.file_uploader("Upload the file", type="bin")

    # Graph selection section
    st.sidebar.title("Graph Selection")
    graph_options = ['Overview','Data','Invariant Mass', 'Transverse Momentum', 'Pseudorapidity','Total momentum','Transverse momentum of first muon','Transverse momentum of second muon','Best curve fitting for thr first peak','background estimation for the Muon Pair','Full fit',"Predict Transverse Muon Pair Momentum"]
    selected_graph = st.sidebar.selectbox("Select a graph", graph_options)




    # Import data and create the pandas data frame
    if file is not None:

        # Slider for controlling the number of muons
        data = file.read()
        datalist = np.frombuffer(data, dtype=np.float32)
        nevent = int(len(datalist) / 6)
        xdata = np.split(datalist, nevent)
        cols = ["InvMass", "TransMomPair", "PseudoRapid", "MomPair", "TransMom1", "TransMom2"]
        df = pd.DataFrame(xdata, columns=cols)
        # Display the DataFrame on a separate page


        num_muons = st.slider("Number of Muons", min_value=1, max_value=df.shape[0], value=10, step=10)



        df=df.head(num_muons)

        
        

        if selected_graph == 'Data':
            if st.sidebar.radio("Show DataFrame", [False, True]):
                st.title("Data Frame")
                st.dataframe(df)
                st.write("Number of Enteries:", df.shape[0])
        
        
        elif selected_graph == 'Invariant Mass':
            st.sidebar.title("Interactivity")
            interactiveness_options={"Interactive", "Non-Interactive"}
            selected_option= st.sidebar.radio("Select an option", interactiveness_options)
            # Plot histogram of the data
            if selected_option == 'Non-Interactive':
                fig, ax = plt.subplots()
                n, bins, patches = ax.hist(df['InvMass'], bins=1000, color='thistle', density=True)
                ax.set_xlabel('Mass (GeV/c^2)')
                ax.set_ylabel('Candidates')
                ax.set_title('Muon Pair Invariant Mass')

                st.pyplot(fig)
            elif selected_option == 'Interactive':
                p = figure(title='Muon Pair Invariant Mass', x_axis_label='Mass (GeV/c^2)', y_axis_label='Candidates')
                hist, edges = np.histogram(df['InvMass'], bins=1000, density=True)
                source = ColumnDataSource(data=dict(hist=hist, left=edges[:-1], right=edges[1:]))
                p.quad(top='hist', bottom=0, left='left', right='right', source=source, color='thistle')

                # Add hover tool to display coordinates
                hover = HoverTool(tooltips=[('Mass', '@left{0.00} - @right{0.00} GeV/c^2'), ('Candidates', '@hist')])
                p.add_tools(hover)

                # Create an interactive plot using Streamlit
                st.bokeh_chart(p)

            
        elif selected_graph == 'Transverse Momentum':
            # Cut the data depending on Transverse Momentum
            df = df[(df['TransMomPair'] >= 0) & (df['TransMomPair'] <= 15)]

            # Sideband subtraction for Transverse Momentum
            peak1 = df[(df['InvMass'] >= 9.3) & (df['InvMass'] <= 9.6)]
            s1 = df[(df['InvMass'] >= 9.15) & (df['InvMass'] <= 9.3)]
            s2 = df[(df['InvMass'] >= 9.6) & (df['InvMass'] <= 9.75)]

            # Create a 2D plot of Transverse Momentum against Invariant Mass()
            fig1,ax1=plt.subplots()
            hist2d, xedges, yedges, im = ax1.hist2d(peak1['InvMass'], peak1['TransMomPair'], bins=100, cmap=plt.cm.CMRmap)
            ax1.set_ylabel("Transverse Momentum of Muon Pair [GeV/c]")
            ax1.set_xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
            plt.colorbar(im,ax=ax1)
            st.pyplot(fig1)

            fig2,ax2=plt.subplots()
            obs = ax2.hist(peak1['TransMomPair'], bins=20, alpha=0.5,label="obs")
            bck1 = ax2.hist(s1['TransMomPair'], bins=20, alpha=0.5,label="bck1")
            bck2 = ax2.hist(s2['TransMomPair'], bins=20, alpha=0.5,label="bck2")

            n = obs[0] - bck1[0] - bck2[0]

            # Plot signal, background, and the original distribution as a function of transverse momentum
            ax2.plot(obs[1][1:], n, 'bx', label='signal')
            ax2.plot(obs[1][1:], bck1[0] + bck2[0], 'rx', label='Bck')
            ax2.plot(obs[1][1:], obs[0], 'gx', label='Data')
            ax2.set_xlabel('Muon Pair Transverse Momentum (GeV/c)')
            ax2.set_ylabel('Candidates')
            ax2.legend()
            st.pyplot(fig2)

        elif selected_graph == 'Pseudorapidity':
            # Cut the data depending on Pseudorapidity
            df = df[(df['PseudoRapid'] >= 2) & (df['PseudoRapid'] <= 6)]

            # Sideband subtraction for Pseudorapidity
            peak1 = df[(df['InvMass'] >= 9.3) & (df['InvMass'] <= 9.6)]
            s1 = df[(df['InvMass'] >= 9.15) & (df['InvMass'] <= 9.3)]
            s2 = df[(df['InvMass'] >= 9.6) & (df['InvMass'] <= 9.75)]

            # Create a 2D plot of Pseudorapidity against Invariant Mass
            fig3,ax3=plt.subplots()
            hist2d, xedges, yedges, im=ax3.hist2d(peak1['InvMass'], peak1['PseudoRapid'], bins=100, cmap=plt.cm.CMRmap)
            ax3.set_ylabel("Pseudorapidity of Muon Pair")
            ax3.set_xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
            plt.colorbar(im,ax=ax3)
            st.pyplot(fig3)
            fig4,ax4=plt.subplots()
            obs = ax4.hist(peak1['PseudoRapid'], bins=20, alpha=0.5)
            bck1 = ax4.hist(s1['PseudoRapid'], bins=20, alpha=0.5)
            bck2 = ax4.hist(s2['PseudoRapid'], bins=20, alpha=0.5)

            n = obs[0] - bck1[0] - bck2[0]

            # Plot signal, background, and the original distribution as a function of pseudorapidity
            ax4.plot(obs[1][1:], n, 'bx', label='signal')
            ax4.plot(obs[1][1:], bck1[0] + bck2[0], 'rx', label='Bck')
            ax4.plot(obs[1][1:], obs[0], 'gx', label='Data')
            ax4.set_xlabel('Pseudorapidity (GeV/c)')
            ax4.set_ylabel('Candidates')
            ax4.legend()
            st.pyplot(fig4)

        elif selected_graph == 'Total momentum':
            # Identical process as above, however for the total momentum of the muon pair
            df = df[(df['MomPair'] >= 0) & (df['MomPair'] <= 300)]

            peak1 = df[(df['InvMass'] >= 9.3) & (df['InvMass'] <= 9.6)]
            s1 = df[(df['InvMass'] >= 9.15) & (df['InvMass'] <= 9.3)]
            s2 = df[(df['InvMass'] >= 9.6) & (df['InvMass'] <= 9.75)]

            fig5,ax5=plt.subplots()

            hist2d, xedges, yedges, im=ax5.hist2d(peak1['InvMass'], peak1['MomPair'], bins=100, cmap=plt.cm.CMRmap)
            ax5.set_ylabel("(Total) Momentum of Muon Pair [GeV/c]")
            ax5.set_xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
            plt.colorbar(im,ax=ax5)
            st.pyplot(fig5)

            fig6,ax6=plt.subplots()
            obs = ax6.hist(peak1['MomPair'], bins=20, alpha=0.5,label="obs")
            bck1 = ax6.hist(s1['MomPair'], bins=20, alpha=0.5,label="bck1")
            bck2 = ax6.hist(s2['MomPair'], bins=20, alpha=0.5,label="bck2")

            n = obs[0] - bck1[0] - bck2[0]

            ax6.plot(obs[1][1:], n, 'bx', label='signal')
            ax6.plot(obs[1][1:], bck1[0] + bck2[0], 'rx', label='Bck')
            ax6.plot(obs[1][1:], obs[0], 'gx', label='Data')
            ax6.set_xlabel('(Total) Momentum of Muon Pair (GeV/c)')
            ax6.set_ylabel('Candidates')
            ax6.legend()
            st.pyplot(plt)

        elif selected_graph == 'Transverse momentum of first muon':
            # Cut the data depending on the transverse momentum of the first muon
            df = df[(df['TransMom1'] >= 1) & (df['TransMom1'] <= 10)]

            # Sideband subtraction for the transverse momentum of the first muon
            peak1 = df[(df['InvMass'] >= 9.3) & (df['InvMass'] <= 9.6)]
            s1 = df[(df['InvMass'] >= 9.15) & (df['InvMass'] <= 9.3)]
            s2 = df[(df['InvMass'] >= 9.6) & (df['InvMass'] <= 9.75)]

            # Create a 2D plot of the transverse momentum of the first muon against invariant mass
            fig1,ax1=plt.subplots()
            hist2d, xedges, yedges, im = ax1.hist2d(peak1['InvMass'], peak1['TransMom1'], bins=100, cmap=plt.cm.CMRmap)
            ax1.set_ylabel("Transverse Momentum of First Muon [GeV/c]")
            ax1.set_xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
            plt.colorbar(im,ax=ax1)
            st.pyplot(plt)

            fig2,ax2=plt.subplots()
            obs = ax2.hist(peak1['TransMom1'], bins=20, alpha=0.5,label="obs")
            bck1 = ax2.hist(s1['TransMom1'], bins=20, alpha=0.5,label="bck1")
            bck2 = ax2.hist(s2['TransMom1'], bins=20, alpha=0.5,label="bck2")

            n = obs[0] - bck1[0] - bck2[0]

            ax2.plot(obs[1][1:], n, 'bx', label='signal')
            ax2.plot(obs[1][1:], bck1[0] + bck2[0], 'rx', label= 'Bck')
            ax2.plot(obs[1][1:], obs[0], 'gx', label = 'Data')
            ax2.set_xlabel('Transverse Momentum of First Muon [GeV/c]')
            ax2.set_ylabel('Candidates')
            ax2.legend()
            st.pyplot(plt)

        elif selected_graph == 'Transverse momentum of second muon':
            # Cut the data depending on the transverse momentum of the second muon
            df = df[(df['TransMom2'] >= 1) & (df['TransMom2'] <= 10)]

            # Sideband subtraction for the transverse momentum of the second muon
            peak1 = df[(df['InvMass'] >= 9.3) & (df['InvMass'] <= 9.6)]
            s1 = df[(df['InvMass'] >= 9.15) & (df['InvMass'] <= 9.3)]
            s2 = df[(df['InvMass'] >= 9.6) & (df['InvMass'] <= 9.75)]
            fig1,ax1=plt.subplots()
            # Create a 2D plot of the transverse momentum of the second muon against invariant mass
            heatmap=ax1.hist2d(peak1['InvMass'], peak1['TransMom2'], bins=100, cmap=plt.cm.CMRmap)
            ax1.set_ylabel("Transverse Momentum of Second Muon [GeV/c]")
            ax1.set_xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
            cbar = fig1.colorbar(heatmap[3], ax=ax1)
            st.pyplot(fig1)
            fig2,ax2=plt.subplots()


            obs = ax2.hist(peak1['TransMom2'], bins=20, alpha=0.5)
            bck1 = ax2.hist(s1['TransMom2'], bins=20, alpha=0.5)
            bck2 = ax2.hist(s2['TransMom2'], bins=20, alpha=0.5)

            n = obs[0] - bck1[0] - bck2[0]

            ax2.plot(obs[1][1:], n, 'bx', label='signal')
            ax2.plot(obs[1][1:], bck1[0] + bck2[0], 'rx', label= 'Bck')
            ax2.plot(obs[1][1:], obs[0], 'gx', label = 'Data')
            ax2.set_xlabel('Transverse Momentum of Second Muon [GeV/c]')
            ax2.set_ylabel('Candidates')
            ax2.legend()
            st.pyplot(fig2)

        peak1 = df[ (df['InvMass']>=9) & (df['InvMass']<=9.7) ]
        def pdf_comb(x, F, mu, sigma, tau):
        
            """
            Probability distribution function (pdf) of a gaussian and exponential 
            """
            
            a = 9
            b = 9.7
            
            gaussian = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(x-mu)**2 / (2*sigma**2)) 
            exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

            
            return F*(exponential) + (1-F)* ( gaussian )
        
        def likelihood(F, mu, sigma, tau):
        
            """
            Likelihood function of gaussian and exponential fit
            """
            
            return -np.sum(np.log(pdf_comb(peak1['InvMass'], F, mu, sigma, tau)))
            # Determine optimal parameters for fitting of the first peak
        m_comb = Minuit(likelihood, F=0.6, mu=9.45, sigma=0.3, tau=1.6)

        
        if selected_graph == 'Best curve fitting for thr first peak':

            st.dataframe(peak1)
            

    #Plot the mass histogram, fitted function, and signal distribution

            fig1,ax1=plt.subplots()

            n1, bins1, patches1 = ax1.hist(peak1['InvMass'], bins = 1000, color = 'thistle', density = True, label = 'Mass Histogram')
            st.write(bins1)
            y_fit = pdf_comb(bins1[:-1], *m_comb.values)

            st.pyplot(fig1)
            fig2,ax2=plt.subplots()

            ax2.plot(bins1[:-1], y_fit, color = "darkBlue", label = 'Fitted Function')
            ax2.set_xlabel('Mass (GeV/c^2)')
            ax2.set_ylabel('Candidates')
            ax2.set_title('Muon Pair Invariant Mass for Y(1S) peak')

            gaussian = (1-m_comb.values[0])*(1/(m_comb.values[2]*np.sqrt(2*np.pi))) * np.exp(-(bins1[:-1]-m_comb.values[1])**2 / (2*m_comb.values[2]**2))

            ax2.plot(bins1[:-1], gaussian, color = 'red', linestyle = '--', label = 'Signal Distribution')
            ax2.legend()
            st.pyplot(fig2)


            
        # Cut the data to isolate a region with just background
        bgn = df[(df['InvMass'] >= 10.5)]


        # Define the PDF for background exponential curve
        def pdf_bgn(x, tau):
            a = float(bgn['InvMass'].min())
            b = float(bgn['InvMass'].max())
            return (1 / tau) * np.exp(-x / tau) / (np.exp(-a / tau) - np.exp(-b / tau))
        


            # Define the likelihood function for background exponential fit
        def likelihood_bgn(tau):
            return -np.sum(np.log(pdf_bgn(np.array(bgn['InvMass']), tau)))
        

        m_comb_bgn = Minuit(likelihood_bgn, tau = 1.76)

        m_comb_bgn.migrad()  # run optimiser
        m_comb_bgn.hesse()   # run covariance estimator

        
        if selected_graph =="background estimation for the Muon Pair":
        

            

            # Fit the exponential curve to the background data
            result = minimize(likelihood_bgn, x0=[1.76])
            tau = result.x[0]

            # Create the background exponential curve
            x_vals = np.linspace(bgn['InvMass'].min(), bgn['InvMass'].max(), 100)
            y_vals = pdf_bgn(x_vals, tau)

            # Create a DataFrame for Bokeh plot
            bokeh_data = pd.DataFrame({'x': x_vals, 'y': y_vals})

            # Choose between Bokeh and Matplotlib plots using a radio button
            plot_option = st.sidebar.radio("Select a plot", ('Bokeh', 'Matplotlib'))

            # Plot the selected plot
            if plot_option == 'Bokeh':
                # Load data and perform fitting


                # Perform histogram
                n_bgn, bins_bgn, _ = plt.hist(bgn['InvMass'], bins=1000, color='thistle', density=True, label='Mass Histogram')

                # Perform histogram
                n_bgn, bins_bgn, _ = plt.hist(bgn['InvMass'], bins=1000, color='thistle', density=True, label='Mass Histogram')
                y_bgn = pdf_bgn(bins_bgn[:-1], *m_comb_bgn.values)  # Replace pdf_bgn with your background fitting function

                # Create Bokeh figure
                p = figure(title='Muon Pair Invariant Mass (background)', x_axis_label='Mass (GeV/c^2)', y_axis_label='Candidates')
                p.add_layout(Title(text='Fitted Background Exponential', align='center'), 'above')

                # Plot background histogram
                p.quad(top=n_bgn, bottom=0, left=bins_bgn[:-1], right=bins_bgn[1:], fill_color='thistle', line_color='white')

                # Plot fitted exponential
                x = np.linspace(bins_bgn[0], bins_bgn[-1], 1000)
                y_bgn = pdf_bgn(x, *m_comb_bgn.values)  # Replace pdf_bgn with your background fitting function
                p.line(x, y_bgn, color='darkblue', legend_label='Fitted Background Exponential')
                # Add hover tool to display coordinates
                hover = HoverTool(tooltips=[('Mass', '@left{0.00} - @right{0.00} GeV/c^2'), ('Candidates', '@hist')])
                p.add_tools(hover)


                # Display the plot using Streamlit
                st.bokeh_chart(p)

            else:
                # Create a Matplotlib plot

        
                # Perform histogram
                n_bgn, bins_bgn, patches = plt.hist(bgn['InvMass'], bins=1000, color='thistle', density=True, label='Mass Histogram')
                y_bgn = pdf_bgn(bins_bgn[:-1], *m_comb_bgn.values)  # Replace pdf_bgn with your background fitting function

                # Plot background histogram and fitted exponential
                plt.plot(bins_bgn[:-1], y_bgn, color='darkblue', label='Fitted Background Exponential')
                plt.xlabel('Mass (GeV/c^2)')
                plt.ylabel('Candidates')
                plt.title('Muon Pair Invariant Mass (background)')
                plt.legend()

                # Display the plot
                st.pyplot()

        def pdf_tot2(x, F1, F2, F3, s_1, mu_1, mu_2, mu_3):
        
            """
            pdf function describing the entire mass range of interest. Each of the three Upsilon peaks are described by a 
            gaussian and the background is described by an exponential curve
            
            Free parameters are the Upsilon peak means, and the first peak resolution. The two remaining peak resolutions are related
            to the original peak resolution
            """
            
            a = df['InvMass'].min()
            b = df['InvMass'].max()
            
            tau = m_comb_bgn.values[0]
            s_2 = s_1 * mu_2/mu_1
            s_3 = s_1 * mu_3/mu_1
            
            # one gaussian PDF per mass peak 
            g_1 = (1/(s_1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*s_1**2)) 
            g_2 = (1/(s_2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*s_2**2)) 
            g_3 = (1/(s_3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*s_3**2)) 
            
            # exponential pdf for background    
            exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

            
            return F1*(exponential) + (1 - F1 - F2 - F3)* g_1  + F2*g_2 + F3*g_3

        
        def pdf_tot2_for_plot(x, F1, F2, F3, s_1, mu_1, mu_2, mu_3):
            
            """
            This function is used to provide the signal distribution so that it may be plotted
            """
            
            a = df['InvMass'].min()
            b = df['InvMass'].max()
            
            tau = m_comb_bgn.values[0]
            s_2 = s_1 * mu_2/mu_1
            s_3 = s_1 * mu_3/mu_1
            
            # one gaussian PDF per mass peak 
            g_1 = (1/(s_1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*s_1**2)) 
            g_2 = (1/(s_2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*s_2**2)) 
            g_3 = (1/(s_3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*s_3**2)) 
            
            # exponential pdf for background    
            exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

            
            return (1 - F1 - F2 - F3)* g_1  + F2*g_2 + F3*g_3
        
        
        # combined pdf likelihood
        def likelihood_tot2(F1, F2, F3, s_1, mu_1, mu_2, mu_3):
            
            """
            Likelihood function for the triple gaussian fit 
            """
            
            return -np.sum(np.log(pdf_tot2(df["InvMass"], F1, F2, F3, s_1, mu_1, mu_2, mu_3)))
                            

        #Determine the optimal parameters for the full fit

        m_comb_tot2 = Minuit(likelihood_tot2, F1 = 0.6, F2 = 0.15, F3 = 0.15, s_1 = 0.05, mu_1 = 9.35, mu_2=10, mu_3=10.4)

        m_comb_tot2.migrad()  # run optimiser
        m_comb_tot2.hesse()   # run covariance estimator



        if selected_graph=='Full fit':
            subgraph_options = {'mass histogram, fitted function, and signal distribution','plot the residuals '}
            selected_subgraph=st.sidebar.radio("Select a graph of full fit",subgraph_options)
            if selected_subgraph=='mass histogram, fitted function, and signal distribution':
                
                fig1,ax1=plt.subplots()
                (nfull, binsfull, patchesfull) = ax1.hist(df['InvMass'], bins=1000, color = 'thistle', density = True, label = 'Mass Histogram')
                ax1.set_xlabel('Mass (GeV/c^2)')
                ax1.set_ylabel('Candidates')
                ax1.set_title('Triple Gaussian Fit')

                x = binsfull[:-1]

                y_tot2 = pdf_tot2(x, *m_comb_tot2.values)
                ax1.plot(x, y_tot2, color = 'darkblue', label = 'Fitted Function')

                signl = pdf_tot2_for_plot(x, *m_comb_tot2.values)

                ax1.plot(x, signl, color = 'red', linestyle = '--', label = 'Signal Distribution')
                ax1.legend()
                

                st.pyplot(fig1)

            elif selected_subgraph=='plot the residuals ':
                fig1,ax1=plt.subplots()
                (nfull, binsfull, patchesfull) = ax1.hist(df['InvMass'], bins=1000, color = 'thistle', density = True, label = 'Mass Histogram')

                x = binsfull[:-1]

                y_tot2 = pdf_tot2(x, *m_comb_tot2.values)
                plt.plot(x, y_tot2, color = 'darkblue', label = 'Fitted Function')

                signl = pdf_tot2_for_plot(x, *m_comb_tot2.values)
                residuals = y_tot2 - nfull
                plt.plot(binsfull[:-1], residuals, color = 'slateblue', marker = '.', linestyle = 'None')
                plt.axhline(y=0, color='Grey', linestyle='-')
                plt.xlabel('Mass (GeV/c^2)')
                plt.ylabel('Residuals')
                st.pyplot(plt)


    if selected_graph == "Predict Transverse Muon Pair Momentum":
        import pickle
        
        import streamlit as st
        from sklearn.ensemble import RandomForestRegressor

        # Load the pre-trained ML model
        with open('muon_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        # Streamlit app layout
        st.title('Muon Model App')

        # Input fields for user to enter feature values
        inv_mass = st.number_input('InvMass', value=0.0)
        pseudo_rapid = st.number_input('PseudoRapid', value=0.0)
        mom_pair = st.number_input('MomPair', value=0.0)
        trans_mom1 = st.number_input('TransMom1', value=0.0)
        trans_mom2 = st.number_input('TransMom2', value=0.0)

        # Check if the loaded object is a RandomForestRegressor
        if isinstance(model, RandomForestRegressor):
            # Create feature vector from user input
            features = [[inv_mass, pseudo_rapid, mom_pair, trans_mom1, trans_mom2]]

            # Run inference using the loaded model when a button is clicked
            if st.button('Predict'):
                prediction = model.predict(features)[0]
                st.write(f'Predicted TransMomPair: {prediction}')
        else:
            st.write("Loaded model is not a RandomForestRegressor. Please ensure you're loading the correct model.")


